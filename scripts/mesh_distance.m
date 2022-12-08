newton_pth = '/media/ty/ECB0AB91B0AB60B6/blendering/newton/*.obj';
mfem_pth = '/media/ty/ECB0AB91B0AB60B6/blendering/mfem/*.obj';
wrapd_pth = '/media/ty/ECB0AB91B0AB60B6/blendering/wrapd/*.obj';


% newton_files = dir(newton_pth)
% mfem_files = dir(mfem_pth)
% wrapd_files = dir(wrapd_pth)
% 
% diff = zeros(numel(newton_files),2);
% cnt = 1;
% for i = 1:1:numel(newton_files)
%     f1 = fullfile(newton_files(i).folder,newton_files(i).name);
%     f2 = fullfile(mfem_files(i).folder,mfem_files(i).name);
%     f3 = fullfile(wrapd_files(i).folder,wrapd_files(i).name);
% 
%     obj_gt = readOBJ(f1);
%     obj_mfem = readOBJ(f2);
%     obj_wrapd = readOBJ(f3);
% 
%     diff(cnt,1) = norm(obj_gt-obj_mfem,"inf");
%     diff(cnt,2) = norm(obj_gt-obj_wrapd,"inf");
%     cnt = cnt + 1;
% end
% save("diffinf.mat",'diff');
diff =load('diff2.mat','diff').diff;
cnt = size(diff,1);
figure(1); clf;
semilogy(20:cnt,diff(20:cnt,1));
hold on;
semilogy(20:cnt,diff(20:cnt,2));
legend('MFEM','WRAPD')
title('L_\infty');
xlabel('Iterations');