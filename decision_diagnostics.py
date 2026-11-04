"""Tek bir konumun rover ayak izi ve hareket çevresindeki ret nedenlerini sayar."""

from math import ceil,hypot
import numpy as np


def footprint_checks(point,checks,zone,width,height,radius,dx,dy):
    x,y=point
    rows,columns=zone.shape
    left,top=x-width//2,y-height//2
    right,bottom=left+width,top+height
    x0,x1=max(0,left),min(columns,right)
    y0,y1=max(0,top),min(rows,bottom)
    region=np.s_[y0:y1,x0:x1]
    footprint={name:int(mask[region].sum()) for name,mask in checks.items() if mask[region].any()}
    # EDT hareket testi, hücre köşesi payını da içeren dairesel açıklık kullanır.
    envelope=radius+hypot(dx,dy)/2
    rx,ry=ceil(envelope/dx),ceil(envelope/dy)
    yy,xx=np.mgrid[-ry:ry+1,-rx:rx+1]
    disk=np.hypot(xx*dx,yy*dy)<envelope
    global_x,global_y=xx+x,yy+y
    inside=(global_x>=0)&(global_x<columns)&(global_y>=0)&(global_y<rows)
    valid=disk & inside
    motion={name:int(mask[global_y[valid],global_x[valid]].sum())
            for name,mask in checks.items() if mask[global_y[valid],global_x[valid]].any()}
    return {'footprint_with_margin_px':[left,top,width,height],
            'failed_footprint_constraints':footprint,
            'footprint_outside_raster_cells':int(width*height-(x1-x0)*(y1-y0)),
            'footprint_outside_start_zone_cells':int((~zone[region]).sum()),
            'failed_motion_envelope_constraints':motion,
            'motion_envelope_outside_raster_cells':int((disk & ~inside).sum())}
