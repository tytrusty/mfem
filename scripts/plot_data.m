
sqpf = fopen('conv_sqp.txt','r');
wraf = fopen('conv_wrapd.txt','r');
newton = fopen('conv_newton.txt','r');
formatSpec = '%f';
A = fscanf(sqpf,formatSpec);
B = fscanf(wraf,formatSpec);
C = fscanf(newton,formatSpec);

figure(1);clf; 
loglog(1:18,A(1:18),'LineWidth',1); hold on;
loglog(1:numel(B),B,'LineWidth',1);
loglog(1:numel(C),C,'LineWidth',1);
set(gca, 'YScale', 'log') % But you can explicitly force it to be logarithmic
legend('MFEM','WRAPD','Vanilla FEM')
xlabel('Iterations');
ylabel('ND')


figure(1); clf;
mfem = fopen('mfem_KE_1e12.txt','r');
newton8 = fopen('fem_KE_1e8.txt','r');
%newton9 = fopen('fem_KE_1e9.txt','r');
newton10 = fopen('fem_KE_1e10.txt','r');
newton12 = fopen('fem_KE_1e12.txt','r');
formatSpec = '%f';
A = fscanf(mfem,formatSpec);
B = fscanf(newton8,formatSpec);
%C = fscanf(newton9,formatSpec);
D = fscanf(newton10,formatSpec);
E = fscanf(newton12,formatSpec);
plot(A,'LineWidth',1.5); hold on;
plot(B,'Color',[153/255 0 0],'LineWidth',1.5);
%plot(C);
plot(D,'Color',[1 0 0],'LineWidth',1.5);
plot(E,'Color',[255 102 102]./255,'LineWidth',1.5);
title('KE');
xlabel('Iteration');
legend('MFEM 1e12', 'FEM 1e8','FEM 1e10', 'FEM 1e12');