%this function takes, as input, the output of the vector field fitting phase
%and parameterizes it by solving for a smooth, coordinate aligned piecewise
%linear map 
function dataOut = fitTexCoords3D(dataFit, BETA)
%res is an integer that determines which eigenvector we use for the texture
%coordinate. The higher the value the higher the frequency of the pattern

V = dataFit.V;
T = dataFit.T;
%1. Initialize Variables
v = dataFit.v;
w = dataFit.w;
t = dataFit.t;

%Gradient Operators
nV = size(V, 1);
nT = size(T, 1);

G = grad(V, T);
Gx = G(1:nT, 1:nV);
Gy = G(nT+1:2*nT, 1:nV);
Gz = G(2*nT+1:3*nT, 1:nV);

%clear G;

    %2. Cost Function
    % x = [u_x..... u_y.....] (just do one vector field at a time for now
    %v is the vector field with which we are aligning
    %direction (are we aligning with the x or y coordinate lines
%     function cost = coordAlign(x, v, direction) 
%        
%         %alignment cost 
%         fun = [x(1:nV) x((nV+1):2*nV)];
%         
%         dFx = Gx * fun;
%         dFy = Gy * fun;
%         
%         res_v = v(:,1).*dFx(:, direction) + v(:,2).*dFy(:, direction);
%         
%         cost = 0.5.*sum(res_v.^2);
%         
%     end

%     function cost = distortion(x) 
%         fun = [x(1:nV) x((nV+1):2*nV)];
%         
%         dFx = Gx * fun;
%         dFy = Gy * fun;
%         
%         %[Fxx Fxy; Fyx Fyy]
%         %Fxx*Fxx + Fyx.*Fxy
%         %Fxx.*Fxy + Fyx.*Fyy
%         %Fxy*Fyx + Fyy*Fyy
%         Gxx = dFx(:,1).*dFx(:,1) + dFy(:,1).*dFx(:,2) - 1.0;
%         Gyy = dFy(:,2).*dFy(:,2) + dFx(:,2).*dFy(:,1) - 1.0;
%         Gxy = dFx(:,1).*dFx(:,2) + dFy(:,2).*dFy(:,1);
%         
%         cost = 0.5.*sum(Gxx.^2 + 2.*Gxy.^2 +Gyy.^2);
%         
%     end
% 
%     function [c, ceq] = nlconstraints(x)
%         fun = [x(1:nV) x((nV+1):2*nV)];
%         
%         dFx = Gx * fun;
%         dFy = Gy * fun;
%         
%         c = -sum(dFx.^2, 2) -sum(dFy.^2, 2) + 2;
%         
%         ceq = [];
%     end

%solution via eigen problem
vG = sparse(bsxfun(@times, v(:,1), Gx) + bsxfun(@times, v(:,2), Gy) + bsxfun(@times, v(:,3), Gz));
wG = sparse(bsxfun(@times, w(:,1), Gx) + bsxfun(@times, w(:,2), Gy) + bsxfun(@times, w(:,3), Gz));
tG = sparse(bsxfun(@times, t(:,1), Gx) + bsxfun(@times, t(:,2), Gy) + bsxfun(@times, t(:,3), Gz));

%zb = sparse(zeros(size(vG)));
zb = sparse(size(vG,1), size(vG,2));
zv = zeros(size(vG,1),1);
ov = ones(size(vG,1),1);

% A = [vG zb zb; zb vG zb; zb zb vG; wG zb zb ; zb wG zb; zb zb wG; tG zb zb; zb tG zb; zb zb tG;];
% b = [ov; zv; zv; zv; ov; zv; zv; zv; ov];

Aeq = sparse([zb vG zb; zb zb vG; wG zb zb; zb zb wG; tG zb zb; zb tG zb; 1 zb(1, 1:end-1) zb(1,:) zb(1,:); zb(1,:) 1 zb(1,1:end-1) zb(1,:); zb(1,:) zb(1,:) 1 zb(1,1:end-1)]);
beq = [zv; zv; zv; zv; zv; zv; 0; 0; 0];
Aopt = sparse([vG zb zb; zb wG zb; zb zb tG]);
bopt = [ov; ov; ov];

% alpha = 1.0;

COORDSTEST = quadprog(BETA*(Aopt'*Aopt) + Aeq'*Aeq, -BETA.*Aopt'*bopt - Aeq'*beq, [],[], [],[]);
%regularized solve which allows magnitude of field to drift a bit (might
%help with better fitting)
%lambda = 10;
%B = diag(b);proc
%S12 = [zeros(size(b
%S23 = []
%H = [A'*A -A'*B; -B'*A, eye(numel(b)).*lambda];
%f = [zeros(size(A,2),1);-lambda.*ones(numel(b),1)];

% A = [vG zeros(size(wG)); zeros(size(vG)) wG];
%need something to force this to be as linear as poss

% [COORDSX,DX] = eig(full(vG'*vG));
% [COORDSY,DY] = eig(full(wG'*wG));
% [COORDSZ,DZ] = eig(full(tG'*tG));
%COORDSTEST = [COORDSX; COORDSY; COORDSZ];

% COORDSTEST = lsqlin(sparse(A),b);


% numS = numel(b);
% numX = size(A,2);
% ob = eye(numS);
% B = diag(b);
% weight = 2;
% H = sparse([A'*A -A'*B; -B'*A weight*ob]);
% f = [zeros(numX,1); weight*ones(numS,1)];
% %Jcon = [A, -B];
% 
%     function [c,g] = costFunc(x) 
%         c =  0.5*x'*H*x - f'*x;
%         
%          if nargout > 1 % gradient required
%             g = H*x -f;
%         end
%         
%     end

%options = optimoptions('fminunc', 'GradObj', 'On', 'Display', 'iter', 'hessUpdate', 'bfgs', 'algorithm' ,'quasi-newton');
%COORDSTEST = fminunc(@(x)costFunc(x), zeros(numX+numS,1), options);

%try something

%xs = H\f;
%COORDSTEST = xs(1:size(A,1));
%3. Constraints to anchor everything
% fix a single point to the origin 
% tex coords all > 0v(

% lb = zeros(2*nV, 1); 3
% Aeq = zeros(2, 2*nV);
% beq = zeros(size(Aeq,1),1);
% 
% A(1,1) = 1;
% A(2,nV+1) = 1;
% 
% %cost function
% costFunction = @(y) coordAlign(y, v, 1) + coordAlign(y, w, 2) + 0.*distortion(y);
% %4  Solve
% x = zeros(2*nV,1);
% options = optimset('Display', 'iter-detailed', 'Algorithm', 'interior-point',...
%         'MaxFunEvals', inf, 'MaxIter', 5000, 'Hessian', 'lbfgs');
% %xOpt = fmincon(costFunction, x, [], [], Aeq, beq, lb, [], @nlconstraints, options);
%     
% %5  Save results and return
dataOut = dataFit;
dataOut.V = V;
% dataOut.F = T;
dataOut.T = T;
%dataOut.u = [xOpt(1:nV) xOpt((1+nV):2*nV)];
dataOut.u = [COORDSTEST(1:nV) COORDSTEST((1+nV):2*nV) COORDSTEST((1+2*nV):3*nV)];
dataOut.u = matrixnormalize(dataOut.u);
dataOut.Beta = BETA;
%need a term to control the distortion 

end

