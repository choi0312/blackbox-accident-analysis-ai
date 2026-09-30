# SPDX-License-Identifier: GPL-3.0-only
"""Fixed physics ablations on cached RAFT correspondences (no learned state)."""
import numpy as np,cv2
from scipy.ndimage import gaussian_filter1d,median_filter

def smooth(v,s=4):return gaussian_filter1d(median_filter(np.asarray(v),size=5,mode='nearest'),s,mode='nearest')
def basis(p,w,h,horizon):
 f=.85*w;x=(p[:,0]-.5*w)/f;y=(p[:,1]-horizon*h)/f
 u=np.stack([x*y,-(1+x*x),x*y,y,-y],1)
 v=np.stack([y*y,-x*y,1+y*y,-x,np.zeros_like(x)],1)
 return np.stack([u,v],1)*f

def fit(p,flow,w,h,horizon=.45,dim=5):
 b=basis(p+.5*flow,w,h,horizon)[:,:,:dim];a=b.reshape(-1,dim);y=flow.ravel();weights=np.ones(len(p));coef=np.zeros(dim)
 for _ in range(4):
  sw=np.sqrt(np.repeat(weights,2));coef=np.linalg.lstsq(a*sw[:,None],y*sw,rcond=1e-5)[0]
  residual=np.linalg.norm(np.einsum('nij,j->ni',b,coef)-flow,axis=1)
  scale=max(.15,1.4826*np.median(np.abs(residual-np.median(residual))))
  weights=1/(1+(residual/(2.5*scale))**2)
 return np.r_[coef[:2],np.median(residual),np.mean(weights>.25)]


def lane_features(mask,horizon=.45):
 h,w=mask.shape;tracks=[[],[]];last=[None,None]
 for yy in np.linspace(.71*h,.48*h,30):
  y=int(round(yy));bits=mask[max(0,y-1):y+2].max(0);ix=np.flatnonzero(bits);runs=np.split(ix,np.flatnonzero(np.diff(ix)>2)+1);xs=np.array([np.mean(a) for a in runs if len(a)])
  for side in range(2):
   if last[side] is None:
    candidates=xs[xs<.5*w] if side==0 else xs[xs>.5*w]
    if not len(candidates):continue
    x=candidates[np.argmin(abs(candidates-.5*w))]
   else:
    if not len(xs):continue
    oldx,oldy=last[side];expected=.5*w+(oldx-.5*w)*(yy-horizon*h)/(oldy-horizon*h)
    x=xs[np.argmin(abs(xs-expected))]
    if abs(x-expected)>.09*w:continue
   tracks[side].append((x,yy));last[side]=(x,yy)
 out=[]
 for pts in tracks:
  if len(pts)<8:out.extend([0,0,0,0]);continue
  p=np.array(pts);den=p[:,1]-horizon*h;z=.85*w/den;x=(p[:,0]-.5*w)/den
  # Road-plane coordinates in units of camera height; no claim of true intrinsics.
  coef=np.polyfit(z,x,2);err=np.median(abs(np.polyval(coef,z)-x))
  out.extend([float(coef[0]),float(coef[1]),float(err),len(pts)/30])
 return out

