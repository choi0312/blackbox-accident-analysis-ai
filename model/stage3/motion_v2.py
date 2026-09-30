# SPDX-License-Identifier: GPL-3.0-only
"""Frozen multi-rate motion and lane/rotation classification, one video at a time.

No annotations, dataset identities, online fitting, or network are used here.
GPL command-line program, separate from Stage 1/2; YOLOPv2 MIT weights only are
shared with Stage 2. No Stage 2 code is imported.
"""
from pathlib import Path
from contextlib import nullcontext
import json
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from .neural_flow import LocalRaft
from .flexinet_model import FlexiNet
from .motion_geometry import fit, smooth, lane_features
from .gpl_program import _viterbi, VIDEO_SUFFIXES, ACCEL_CLASSES, STEER_CLASSES


def sigmoid(x):
    return 1/(1+np.exp(-np.clip(x,-35,35)))


def acceleration_features(speed, geometry, sigma):
    """Build six low-capacity motion descriptors at one smoothing scale."""
    v=np.clip(speed,0,70);sm=smooth(v,4)
    a=np.gradient(smooth(v,sigma))/.1 if len(v)>1 else np.zeros(1)
    return np.c_[sm,a,a/np.maximum(sm,1),smooth(geometry[:,4],2),
                 smooth(geometry[:,3],3),smooth(abs(v-sm),10)]


def physical(x):
    """Map speed/derivative proxies to a fixed, non-learned class prior."""
    v,a,rel,q,c,noise=x.T
    move=np.maximum(sigmoid((v-.4)/.2),sigmoid((q-.28)/.12))
    threshold=.32+np.minimum(.8,.12*noise)
    up=sigmoid((a-threshold)/.2);down=sigmoid((-a-threshold)/.2)
    p=np.c_[up*move,down*move,np.clip(1-up-down,.01,1)*move,1-move]
    return p/p.sum(1,keepdims=True)


def steering_features(geometry, lanes):
    """Combine robust yaw with confidence-weighted left/right lane curvature."""
    f=lanes['features'];n=int(lanes['n']);pos=lanes['positions']
    a=f[:,0];b=f[:,4]
    wa=f[:,3]*np.exp(-f[:,2]/.025);wb=f[:,7]*np.exp(-f[:,6]/.025)
    curve=(a*wa+b*wb)/np.maximum(wa+wb,1e-5)
    confidence=np.minimum(f[:,3],f[:,7])*np.exp(-abs(a-b)/.003)*np.exp(-(f[:,2]+f[:,6])/.08)
    curve=smooth(np.interp(np.arange(n),pos,curve),8)
    confidence=smooth(np.interp(np.arange(n),pos,confidence),8)
    curve=curve*confidence
    speed=np.maximum(smooth(np.clip(geometry[:,0],0,70),5),0)
    yaw=smooth(np.clip(geometry[:,1],-.7,.7),5)
    return np.c_[yaw,curve,abs(curve),confidence,speed,smooth(geometry[:,4],2)]


def probability(x,head,classes):
    """Evaluate a serialized standardized multinomial logistic-regression head."""
    z=(x-np.asarray(head['mean']))/np.asarray(head['scale'])
    logits=z@np.asarray(head['coef']).T+np.asarray(head['intercept'])
    logits-=logits.max(1,keepdims=True)
    p=np.exp(logits);p/=p.sum(1,keepdims=True)
    result=np.zeros((len(x),len(classes)))
    for j,c in enumerate(head['classes']):result[:,classes.index(c)]=p[:,j]
    return result


def classify(views,head):
    """Fuse original/reflected views and learned/physical acceleration evidence."""
    accel=[]
    for h in head['accel']:
        x=[];p=[]
        for v in views:
            descriptor=acceleration_features(v['cache'][h['key']],v['geometry'],h['sigma'])
            x.append(descriptor);p.append(physical(descriptor))
        x=np.mean(x,0);p=np.mean(p,0)
        q=probability(x,h,list(ACCEL_CLASSES))
        if h['moving']:
            q[:,:3]*=1-p[:,3,None];q[:,3]=p[:,3]
        weight=head['accel_physical_weight']
        accel.append(h['weight']*((1-weight)*q+weight*p))
    xa=steering_features(views[0]['geometry'],views[0]['lanes'])
    xb=steering_features(views[1]['geometry'],views[1]['lanes'])
    xb[:,:2]*=-1
    steering=np.mean([probability((xa+xb)/2,h,list(STEER_CLASSES)) for h in head['steer']],0)
    return np.sum(accel,0),steering


class FrozenMotion:
    """Own all frozen Stage 3 feature extractors for one streaming pipeline."""

    def __init__(self,root):
        self.flow=LocalRaft(root/'raft_large.pth',width=384,updates=8)
        self.device=self.flow.device
        self.flex=FlexiNet()
        state=torch.load(root/'flexinet_kitti.pth',map_location='cpu',weights_only=True)
        state=state.get('state_dict',state)
        self.flex.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
        self.flex.to(self.device).eval().requires_grad_(False)
        self.segment=torch.jit.load(str(root.parent/'stage2/yolopv2.pt'),map_location='cpu')
        self.segment.to(self.device).eval()
        for parameter in self.segment.parameters():parameter.requires_grad_(False)
        if self.device.type=='cuda':self.segment.half()
        for network in [self.flow.model,self.flex,self.segment]:
            assert all(not p.requires_grad for p in network.parameters())
            assert all(not module.training for module in network.modules())

    @torch.inference_mode()
    def lane(self,rgb):
        h,w=rgb.shape[:2];r=640/max(h,w);nw,nh=round(w*r),round(h*r)
        dw,dh=(640-nw)%32,(640-nh)%32;l,t=dw//2,dh//2
        x=cv2.copyMakeBorder(cv2.resize(rgb,(nw,nh)),t,dh-t,l,dw-l,cv2.BORDER_CONSTANT,value=(114,114,114))
        tensor=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).to(self.device)
        tensor=tensor.half() if self.device.type=='cuda' else tensor.float()
        _,_,lane=self.segment(tensor[None]/255)
        ah,aw=lane.shape[-2:];ph,pw=x.shape[:2]
        lane=lane[...,round(t*ah/ph):round((t+nh)*ah/ph),round(l*aw/pw):round((l+nw)*aw/pw)]
        mask=(lane[0,0]>.5).cpu().numpy().astype('uint8')
        mask=cv2.resize(mask,(384,h),interpolation=cv2.INTER_NEAREST)
        return lane_features(mask)

    def motion(self,first,second):
        h,w=first.shape[:2];yy,xx=np.mgrid[3:h:6,3:w:6]
        p=np.stack([xx,yy],-1).reshape(-1,2).astype(float)
        roi=(p[:,1]/h>.51)&(p[:,1]/h<.71)&(abs(p[:,0]/w-.5)<(.07+(p[:,1]/h-.51)*1.25))
        # Quantize the flow descriptor to the fixed inference representation.
        flow=self.flow(first,second)[yy,xx].reshape(-1,2).astype('float16').astype(float)
        v=flow[roi];values=fit(p[roi],v,w,h,.45,5)
        values[0]*=1.45/.1;values[1]/=.1
        mag=np.linalg.norm(v,axis=1)*512/w
        return np.r_[values,np.quantile(mag,.25),np.median(mag)]

    @torch.inference_mode()
    def speeds(self,gray):
        n=len(gray);pos=np.unique(np.r_[np.arange(0,n,3),n-1]).astype(int)
        x=(gray.astype(np.float32)/255)[:,None]*2-1
        out={}
        for step in [1,5]:
            offsets=(np.arange(13)-6.7777778)*step;pred=[]
            for j in range(0,len(pos),8):
                indices=np.clip(np.rint(pos[j:j+8,None]+offsets).astype(int),0,n-1)
                amp=torch.autocast('cuda',dtype=torch.float16) if self.device.type=='cuda' else nullcontext()
                with amp:
                    y=self.flex(torch.from_numpy(x[indices]).to(self.device)).float().ravel()
                pred.extend(y.cpu().tolist())
            out[f'raw{step}']=np.interp(np.arange(n),pos,pred)
        return out

    def extract(self,path):
        """Extract two reflection-consistent views from every decoded frame.

        Stride is always one. Only two RGB frames are retained at once; the full
        video is stored only as compact 64x64 grayscale samples and descriptors.
        """
        cap=cv2.VideoCapture(str(path))
        if not cap.isOpened():raise ValueError(f'Cannot decode Stage 3 video: {path}')
        small=[[],[]];lanes=[[],[]];motions=[[],[]];lane_pos=[];flow_time=[]
        previous=None;penultimate=None;i=0;size=None
        try:
            while True:
                ok,bgr=cap.read()
                if not ok:break
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                if size is None:size=(384,max(128,round(384*rgb.shape[0]/rgb.shape[1]/8)*8))
                current=[]
                for flip in [0,1]:
                    image=np.ascontiguousarray(rgb[:,::-1]) if flip else rgb
                    small[flip].append(np.asarray(Image.fromarray(image).convert('L').resize((64,64),Image.Resampling.BILINEAR)))
                    current.append(cv2.resize(image,size))
                if i%5==0:
                    lane_pos.append(i)
                    for flip in [0,1]:lanes[flip].append(self.lane(current[flip]))
                if i%5==1:
                    flow_time.append(i-.5)
                    for flip in [0,1]:motions[flip].append(self.motion(previous[flip],current[flip]))
                penultimate=previous;previous=current;i+=1
        finally:cap.release()
        n=i
        if n==0:raise ValueError(f'Empty Stage 3 video: {path}')
        if lane_pos[-1]!=n-1:
            lane_pos.append(n-1)
            for flip in [0,1]:lanes[flip].append(self.lane(previous[flip]))
        # Use a backward adjacent pair so the final frame also has a motion sample.
        if n>1:
            flow_time.append(n-1.5)
            for flip in [0,1]:motions[flip].append(self.motion(penultimate[flip],previous[flip]))
        else:
            flow_time=[0.]
            for flip in [0,1]:motions[flip]=[np.array([0,0,0,1,0,0.])]
        views=[]
        for flip in [0,1]:
            raw=np.stack(motions[flip])
            geometry=np.stack([np.interp(np.arange(n),flow_time,raw[:,j]) for j in range(6)],1).astype(np.float32)
            views.append({'cache':self.speeds(np.stack(small[flip])), 'geometry':geometry,
                          'lanes':{'features':np.array(lanes[flip]),'positions':np.array(lane_pos),'n':n}})
        return views


def predict(data_dir,model_dir):
    """Decode each video independently and emit one smoothed label pair per frame."""
    base=Path(data_dir);directory=base/'videos' if (base/'videos').is_dir() else base
    paths=sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    columns=['ID','sample_index','accel_label','steer_label']
    if not paths:return pd.DataFrame(columns=columns)
    if len({p.stem for p in paths})!=len(paths):raise ValueError('Duplicate Stage 3 IDs')
    cv2.setNumThreads(2);root=Path(model_dir)
    head=json.loads((root/'motion_v2_head.json').read_text())
    if head.get('schema')!=1 or head['classes_accel']!=list(ACCEL_CLASSES) or head['classes_steer']!=list(STEER_CLASSES):raise ValueError('Invalid frozen head schema')
    engine=FrozenMotion(root);rows=[]
    for path in paths:
        views=engine.extract(path);pa,ps=classify(views,head)
        a=_viterbi(np.log(np.maximum(pa,1e-7)),head['transition_accel'])
        s=_viterbi(np.log(np.maximum(ps,1e-7)),head['transition_steer'])
        rows.extend((path.stem,i,ACCEL_CLASSES[aa],STEER_CLASSES[ss]) for i,(aa,ss) in enumerate(zip(a,s)))
    return pd.DataFrame(rows,columns=columns)
