q = sym('q',[9 1]);
Dm = sym('Dm',[2 2]);
assume(q,'real')
assume(Dm,'real')

Ds = [q(4:6)-q(1:3) q(7:9)-q(1:3)];
F=Ds*Dm
ccode(jacobian(F(:),q))


