function [ans1,ans2,ans3] = dy(CSI)

mm = ones(1,114);
csia_1 = abs(CSI(1,:));
csia_2 = abs(CSI(2,:));
csia_3 = abs(CSI(3,:));
               
xxx1= csia_1.^2;  
en1 = dot(xxx1,mm)/(norm(xxx1)*norm(mm));
xxx2= csia_2.^2;
en2 = dot(xxx2,mm)/(norm(xxx2)*norm(mm));
xxx3= csia_3.^2;  
en3 = dot(xxx3,mm)/(norm(xxx3)*norm(mm));
ans1 = acos(en1)*180/pi;
ans2 = acos(en2)*180/pi;
ans3 = acos(en3)*180/pi;

end

