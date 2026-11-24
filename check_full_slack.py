"""Check the 40- and 79-term factorizations on all vertices of the actual polytope."""
from argparse import ArgumentParser
from itertools import product
import json
from pathlib import Path

import numpy as np


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("polytope",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.polytope.read_text(encoding="utf-8"))
    dtype=np.int64
    points=np.array(data["candidate_points_y"],dtype=dtype)
    rows=data["facets_of_base"]
    retained=sorted((f for f in rows if f["coordinate_row"] is not None),key=lambda f:f["coordinate_row"])
    split=[f for f in rows if f["coordinate_row"] is None]
    record=retained+split
    a=np.array([f["a"] for f in record],dtype=dtype)
    b=np.array([f["b"] for f in record],dtype=dtype)
    base_slack=b[:,None]-a@points.T
    edges=data["edge_endpoints"]
    R=np.array([[int(v in e) for v in range(12)] for e in edges],dtype=dtype)
    A=np.array([[int(e//4==2*t) for t in range(3)] for e in range(24)],dtype=dtype)
    S=[];U=[];selectors=[];coords=[]
    for i,point in enumerate(points):
        t,node=divmod(i,13)
        upper={}
        for s,cluster in [((t+1)%3,(2*t+4)%6),((t+2)%3,(2*t+3)%6)]:
            weights=np.array([int(v//2==cluster) for v in range(12)],dtype=dtype)
            if node:
                weights[node-1]+=data["scale"]
            assert np.array_equal(R@weights+A[:,s],base_slack[:24,i])
            upper[s]=weights
        slack=base_slack[24:,i]
        active=np.flatnonzero(slack)
        for state in product((0,1),repeat=len(active)):
            z=np.zeros(13,dtype=dtype)
            z[active]=np.array(state,dtype=dtype)*slack[active]
            S.append(np.r_[base_slack[:24,i],z,slack-z])
            U.append(upper)
            selectors.append(tuple(upper))
            coords.append(tuple(np.r_[point,z]))
    S=np.column_stack(S)
    assert S.shape==(50,150) and len(set(coords))==150 and S.min()==0
    identity=np.eye(26,dtype=dtype)
    W1=np.zeros((50,40),dtype=dtype)
    W1[:24,:12]=R;W1[:24,12:14]=A[:,:2];W1[24:,14:]=identity
    H1=np.zeros((40,150),dtype=dtype)
    for i in range(150):
        s=next(s for s in (0,1) if s in selectors[i])
        H1[:12,i]=U[i][s]
        H1[12+s,i]=1
    H1[14:]=S[24:]
    assert np.array_equal(W1@H1,S)

    W=np.zeros((100,79),dtype=dtype)
    W[:24,:12]=R;W[50:74,12:24]=R
    W[:24,24:27]=A;W[50:74,24:27]=A
    W[24:50,27:53]=identity;W[74:,53:]=identity
    H=np.zeros((79,150*150),dtype=dtype)
    for col,(i,j) in enumerate(product(range(150),repeat=2)):
        s=next(s for s in selectors[i] if s in selectors[j])
        H[:12,col]=U[i][s];H[12:24,col]=U[j][s];H[24+s,col]=1
        H[27:53,col]=S[24:,i];H[53:,col]=S[24:,j]
    actual=np.vstack([np.repeat(S,150,axis=1),np.tile(S,(1,150))])
    overflow_bound=int(W.max())*int(H.max())*W.shape[1]
    assert overflow_bound<np.iinfo(dtype).max
    assert min(W.min(),H.min(),W1.min(),H1.min())==0
    assert np.array_equal(W@H,actual)
    report=dict(vertices_of_factor=150,facets_of_factor=50,single_NMF_width=40,
                single_entries_checked=S.size,product_vertices=22500,product_facets=100,
                product_NMF_width=79,product_entries_checked=actual.size,
                exact_integer_residual=0,integer_overflow_upper_bound=overflow_bound)
    with args.out.open("x",encoding="utf-8") as fd:
        json.dump(report,fd,indent=2);fd.write("\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
