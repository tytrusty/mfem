ym = readDMAT('../output/dhat_values.dmat');
a = readDMAT('../output/finalconvergence_converge_fem.dmat');
b = readDMAT('../output/finalconvergence_converge_mfem.dmat');
figure(2); clf;
loglog(ym,a,'.-','MarkerSize',15);
hold on;
loglog(ym,b,'.-','MarkerSize',15);
xlabel("dhat");
ylabel("Gradient Norm");
legend("Vanilla IPC", "Mixed IPC");
title("Convergence after 50 iterations")


figure(1); clf;

a = readDMAT('../output/convergence_fem.dmat');
b = readDMAT('../output/convergence_mfem.dmat');
N = numel(ym);
M = size(b,2);

%colormap(CustomColormap);
colormap(winter(1000))
cmap = winter(N); 
hold on;
for i=1:N
    semilogy(1:M,a(i,1:M),'-','Color',cmap(i,:));
    semilogy(1:M,b(i,1:M),'--','Color',cmap(i,:));

    if (i == 1)
        legend("Vanilla IPC", "Ours");
    end
end
set(gca, 'YScale', 'log');
a= colorbar;
a.Label.String = "\hat d";
xlabel("Iterations");
ylabel("||\delta x||");
title("Convergence of Collision Simulation FEM vs MFEM")
