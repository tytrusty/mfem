[V,T,F] = readMESH('beam_5k.mesh');
[F,J,K]=boundary_faces(T);
%tsurf(F,V);

bmin = min(V,[],1);
bmax = max(V,[],1);
center = mean([bmin;bmax],1);

centroids = (V(T(:,1),:) + V(T(:,2),:) + V(T(:,3),:)) ./ 3;

off=[0.2 0.4 0.4];
%id1 = all( (centroids > (center-bmin).*off + bmin) ,2);bone_ids
%id2 = all(centroids < bmax - (bmax-center).*off ,2);

bmax_mid = bmax;
bmax_mid(1) = center(1);
id1 = all( centroids > (bmin + (bmax_mid-bmin).*[0.2 0.4 0.4]),2);
id2 = all( centroids < (bmax_mid - (bmax_mid-bmin).*[0.2 0.4 0.4]),2);
bone_ids = id1&id2;

bmin_mid = bmin;
bmin_mid(1) = center(1);
id1 = all( centroids > (bmin_mid + (bmax-bmin_mid).*[0.2 0.4 0.4]),2);
id2 = all( centroids < (bmax - (bmax-bmin_mid).*[0.2 0.4 0.4]),2);
sum((id1&id2))
bone_ids = bone_ids | (id1&id2);

muscle_ids = ~bone_ids;
all_ids = 1:size(T,1);
 
bone_ids = all_ids(bone_ids)';
muscle_ids = all_ids(muscle_ids)';

writeDMAT('beam_bone_V.dmat',V);
writeDMAT('beam_bone_T.dmat',T-1);
writeDMAT('muscle_tets.dmat',muscle_ids-1);
writeDMAT('bone_tets.dmat',bone_ids-1);
