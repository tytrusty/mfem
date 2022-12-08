d=3;
F=sym('F',[d d]);
s = sym('S',[d*2,1]);
sval = sym('sval',[d,1]);
syms mu la c
assume(F,'real')
assume(s,'real')
assume(mu,'real')
assume(la,'real')
assume(c,'real')
assume(sval,'real')
% 
S = [s(1) s(4) s(5);
     s(4) s(2) s(6);
     s(5) s(6) s(3)];
% S = [s(1) s(3);
%      s(3) s(2)];

% stable neohookean
I3=det(F);
I2=trace(F'*F);
I1=trace(sqrt(F'*F));
% I3=det(S);
% I2=trace(S'*S);
% I2 = sum(sval.^2);
% I3 = prod(sval);


snh= 0.5*mu*(I2-d)- mu*(I3-1)+ 0.5*la*(I3-1)^2;
H=simplify(hessian(snh,s(:)));
g=simplify(gradient(snh,s(:)));
ccode(snh)
ccode(H)
ccode(g)

% Isotropic Fung
E = S'*S - eye(d);
fung = 0.5 * ((mu*(I2 - d)) + la*(exp(c*(I2-d)) - 1));
fung = 0.5*mu*(I2-3) + 0.5*la*(I3-1)^2 + ...
    0.5*mu*(exp(c*0.5*(I2-3))-1) - (mu+mu*c)*(I3-1);
fung = 0.5*mu*(I2-3) + 0.5*la*(I3 - 1 - (mu+mu*c)/la)^2 + ...
    0.5*mu*(exp(c*0.5*(I2-3)) - 1);
H=(hessian(fung,s(:)));
g=gradient(fung,s(:));
ccode(fung)
ccode(H)
ccode(g)

% neohookean
%nh = 0.5*mu*(I2/(I3^(2/3)) - 3) + 0.5*la*(I3-1)^2;
nh = 0.5*mu*(I2- 3) - mu*log(I3) + 0.5*la*(log(I3))^2;
% H=(hessian(nh,s(:)));
% g=gradient(nh,s(:));
H=(hessian(nh,F(:)));
g=gradient(nh,F(:));
ccode(nh)
ccode(H)
ccode(g)

%F=R*S;
%J=det(F);
%I3=trace(F'*F)/J^(2/3);
%snh=0.5*mu*(I3-3)+ 0.5*la*(J-1)^2;

% Corotational material model
arap= mu*0.5*trace( (S - eye(d))*(S - eye(d))');
H=hessian(arap,s(:));
g=gradient(arap,s(:));
ccode(arap)
ccode(H)
ccode(g)

corot = la*0.5*trace(S-eye(d))^2 + 2*arap;
H=hessian(corot,s(:));
g=gradient(corot,s(:));
ccode(corot)
ccode((H))
ccode(g)


fixed_corot = 0.5*la*(I3-1)^2 + 2*arap;
H=hessian(fixed_corot,s(:));
g=gradient(fixed_corot,s(:));
ccode(fixed_corot)
ccode((H))
ccode(g)

fixed_corot = 0.5*la*(I3-1)^2 + mu*(I2 - 2*I1 + d);
H=hessian(fixed_corot,F(:));
g=gradient(fixed_corot,F(:));
ccode(fixed_corot)
ccode((H))
ccode(g)
