function [V,J]=diagonalizeA(A)
    [V,J]=eig(A);
    % Deal with complex solutions, if they happen:
    a=imag(diag(J)); b=real(diag(J));
    if any(abs(a./b)>1e-15) %If there are (truly) complex eigen-values, will transform to the real-jordan form
        [V,J] = cdf2rdf(V,J);
        else %Ignore imaginary parts
        V=real(V);
        J=real(J);
    end
    % Sort states by decay rates: (these are only the decay rates if J is diagonal)
    [~,idx]=sort(diag(J)); %This works if the matrix is diagonalizable
    J=J(idx,idx);
    V=V(:,idx);

    %Alt: use block diagonal Schur
    %[V,J]=bdschur(A);
end
