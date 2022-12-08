% Segment a heterogeneous mesh by providing a set of OBJs embedded
% in the tetmesh representing different materials.
[V,T,F]=readMESH("~/Desktop/models/gummy_bear.mesh");
F = boundary_faces(T);
%tsurf(F,V,'FaceAlpha',0.5);
%axis equal;
[V1,F1] = readOBJ("~/Downloads/gummy_bear_inside.obj");
%hold on;
%tsurf(F1,V1,'FaceAlpha',0.5);

c = (V(T(:,1),:) + V(T(:,2),:) + V(T(:,3),:) + V(T(:,4),:))/4;

I = signed_distance(c,V1,F1);
%c_inside = c(I<0,:);
%plot3(c_inside(:,1),c_inside(:,2),c_inside(:,3),'r.','MarkerSize',15);
ids = zeros(size(T,1),1);
ids(I<0) = 1;
writeDMAT('gummy_bear_material_ids.dmat',ids);