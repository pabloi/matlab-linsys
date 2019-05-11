function [newcPst,newXs,newPt,Ht]=sqrtRTS(cPft,cPst,xp,xf,prevXs,At,cQt)
  %Implements the Rauch-Tung-Striebel backward recursion
  %https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers)
  %Through a square-root approach. See statKalmanSmoother>backStepRTS

  iAt=pinv(At); %This is order-of-magnitude slower than inv(A)
  ey=eye(size(At));
  invertibleA=norm(At*iAt-ey,'fro')<1e-9;

  %To do: consider the case with infinite variances (Pf) as done on statKalmanSmoother>backRTS
  infIdx=isinf(diag(cPft));
  Nx=size(cQt,1);
  if ~any(infIdx) %The usual case: we have a proper/numerically well-conditioned prior from filter
     AcPft=cPft*At;
     %[~,cPpt]=qr([cQt; AcPft],0);
     %Ht=cPpt'\(cPpt\AcPft'*cPft);
     Ht=(AcPft'*AcPft+cQt'*cQt)\AcPft'*cPft;
     M=[cQt zeros(Nx); AcPft cPft; zeros(Nx) cPst*Ht];
     [~,R]=qr(M,0);
     newcPst=R(Nx+1:end,Nx+1:end);
     newPt=(cPst'*cPst)*Ht;
     newXs=xf+Ht'*(prevXs-xp);
  else %This happens when we started filtering from an improper prior
      if invertibleA
          AcPf=A*cPft;
          pp=AcPf*AcPf'+cQt*cQt;
          [icP,~]=pinvchol2(pp); %Handles infinite (and 0 covariances) covariances. Substitutes non-diagonal Inf elements by 0.
          [newPs,newXs,newPt,H]=backStepRTS_invA(icP,cPs,xp,prevXs,cQ,bu,iA);
      else
          error('Unimplemented')
      end
  end
end
