function NNtrain1()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Preparation of dataset

%collection of dataset
A = [1/5 1/10 1/30; 1/10 2/15 1/10;  1/30 1/10 1/5];
i=1;
for n=0.01:0.01:0.5
rhs= [  (pi*pi*n*n+2*cos(pi*n)-2)/(pi*pi*pi*n*n*n);
(-2*pi*n*sin(pi*n)-4*cos(pi*n)+4)/(pi*pi*pi*n*n*n); 
((2-pi*pi*n*n)*cos(pi*n)+2*pi*n*sin(pi*n)-2)/(pi*pi*pi*n*n*n) ];
A;
rhs;
u=A\rhs;
dataset_in(i)=n; 
dataset_u1(i)=u(1); 
dataset_u2(i)=u(2); 
dataset_u3(i)=u(3); 
i=i+1;
end
ndataset=i-1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
a1=1.0; b1=1.0; c1=1.0; d1=1.0;

a2=1.0; b2=1.0; c2=10.0; d2=-1.0;

a3=3.0; b3=3.0; c3=10.0; d3=-1.0;

eta1=0.1;
eta2=0.1;
eta3=0.1;

r = 0 + (1-0).*rand(ndataset,1);
r=r.*ndataset;
for j=1:ndataset
  i=floor(r(j));
  if(i==0)
    i=1;
  end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %1
  a1_=a1; b1_=b1; c1_=c1; d1_=d1;
  %evaluation of the first NN modeling a1
  eval1 = c1*1.0/(1.0+exp(-(a1*dataset_in(i)+b1)))+d1;
  error1 = 0.5*(eval1-dataset_u1(i))^2;
  %a1=a1-eta*derivative of error with respect to a
  %d error/da = d/da [[1/2*(c sigma (ax+b)+d-y(x))^2]=cxe^(-ax-b)(eval1-y(x))/(e^{-ax-b)+1)^2
  derrorda = c1*dataset_in(i)*exp(-a1*dataset_in(i)-b1)*(eval1-dataset_u1(i))/(exp(-a1*dataset_in(i)-b1)+1)^2;
  a1=a1-eta1*  derrorda;
  %b1=b1-eta*derivative of error with respect to b
  %d error/db = d/db [1/2*(c sigma (ax+b)+d-y(x))^2]=ce^(-ax-b)(eval1-y(x))/(e^{-ax-b)+1)^2
  derrordb = c1*exp(-a1_*dataset_in(i)-b1)*(eval1-dataset_u1(i))/(exp(-a1_*dataset_in(i)-b1)+1)^2;
  b1=b1-eta1*  derrordb;
  %c1=c1-eta*derivative of error with respect to c
  %d error/dc = d/dc [1/2*(c sigma (ax+b)+d-y(x))^2]=(eval1-y(x))/(e^{-ax-b)+1)
  derrordc = (eval1-dataset_u1(i))/(exp(-a1*dataset_in(i)-b1)+1);
  c1=c1-eta1*  derrordc;
  %d1=d1-eta*derivative of error with respect to d
  %d error/dd = d/dd [1/2*(c sigma (ax+b)+d-y(x))^2]=eval1-y
  derrordd = (eval1-dataset_u1(i));
  d1=d1-eta1*  derrordd;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %2
  a2_=a1; b2_=b2; c2_=c2; d2_=d2;
  %evaluation of the first NN modeling a2  
  eval2 = c2*1.0/(1.0+exp(-(a2*dataset_in(i)+b2)))+d2;
  error2 = 0.5*(eval2-dataset_u2(i))^2;
  %a2=a2-eta*derivative of error with respect to a
  %d error/da = d/da [[1/2*(c sigma (ax+b)+d-y(x))^2]=cxe^(-ax-b)(eval2-y(x))/(e^{-ax-b)+1)^2
  derrorda = c2*dataset_in(i)*exp(-a2*dataset_in(i)-b2)*(eval2-dataset_u2(i))/(exp(-a2*dataset_in(i)-b2)+1)^2;
  a2=a2-eta2*  derrorda;
  %b2=b2-eta*derivative of error with respect to b
  %d error/db = d/db [1/2*(c sigma (ax+b)+d-y(x))^2]=ce^(-ax-b)(eval2-y(x))/(e^{-ax-b)+1)^2
  derrordb = c2*exp(-a2_*dataset_in(i)-b2)*(eval2-dataset_u2(i))/(exp(-a2_*dataset_in(i)-b2)+1)^2;
  b2=b2-eta2*  derrordb;
  %c2=c2-eta*derivative of error with respect to c
  %d error/dc = d/dc [1/2*(c sigma (ax+b)+d-y(x))^2]=(eval2-y(x))/(e^{-ax-b)+1)
  derrordc = (eval2-dataset_u2(i))/(exp(-a2*dataset_in(i)-b2)+1);
  c2=c2-eta2*  derrordc;
  %d2=d2-eta*derivative of error with respect to d
  %d error/dd = d/dd [1/2*(c sigma (ax+b)+d-y(x))^2]=
  derrordd = (eval2-dataset_u2(i));
  d2=d2-eta2*  derrordd;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  %3
  a3_=a1; b3_=b2; c3_=c2; d3_=d2;
  %evaluation of the first NN modeling a3 
  eval3 = c3*1.0/(1.0+exp(-(a3*dataset_in(i)+b3)))+d3;
  error3 = 0.5*(eval3-dataset_u3(i))^2;
  %a3=a3-eta*derivative of error with respect to a
  %d error/da = d/da [[1/2*(c sigma (ax+b)+d-y(x))^2]=cxe^(-ax-b)(eval3-y(x))/(e^{-ax-b)+1)^2
  derrorda = c3*dataset_in(i)*exp(-a3*dataset_in(i)-b3)*(eval3-dataset_u3(i))/(exp(-a3*dataset_in(i)-b3)+1)^2;
  a3=a3-eta3*  derrorda;
  %b3=b3-eta*derivative of error with respect to b
  %d error/db = d/db [1/2*(c sigma (ax+b)+d-y(x))^2]=ce^(-ax-b)(eval3-y(x))/(e^{-ax-b)+1)^2
  derrordb = c3*exp(-a3_*dataset_in(i)-b3)*(eval3-dataset_u3(i))/(exp(-a3_*dataset_in(i)-b3)+1)^2;
  b3=b3-eta3*  derrordb;
  %c3=c3-eta*derivative of error with respect to c
  %d error/dc = d/dc [1/2*(c sigma (ax+b)+d-y(x))^2]=(eval3-y(x))/(e^{-ax-b)+1)
  derrordc = (eval3-dataset_u3(i))/(exp(-a3*dataset_in(i)-b3)+1);
  c3=c3-eta3*  derrordc;
  %d3=d3-eta*derivative of error with respect to d
  %d error/dd = d/dd [1/2*(c sigma (ax+b)+d-y(x))^2]=eval3-y
  derrordd = (eval3-dataset_u3(i));
  d3=d3-eta3*  derrordd;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%store for making plots later
  y1a(j)=a1;
  y1b(j)=b1;
  y1c(j)=c1;
  y1d(j)=d1;
  y2a(j)=a2;
  y2b(j)=b2;
  y2c(j)=c2;
  y2d(j)=d2;
  y3a(j)=a3;
  y3b(j)=b3;
  y3c(j)=c3;
  y3d(j)=d3;
  yeval1(j)=eval1;
  yeval2(j)=eval2;
  yeval3(j)=eval3;
  yu1(j)=dataset_u1(i);
  yu2(j)=dataset_u2(i);
  yu3(j)=dataset_u3(i);
  ye1(j)=error1;
  ye2(j)=error2;
  ye3(j)=error3;
%
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot convergence
x=1:1:ndataset;
hold on

loglog(x,ye1,'LineWidth',3,x,ye2,'LineWidth',3,x,ye3,'LineWidth',3);
h=legend('error1','error2','error3');
set(h,'FontSize',30);
set(gca,'FontSize',30);

figure

plot(x,y2a,'LineWidth',3,x,y2b,'LineWidth',3,x,y2c,'LineWidth',3,x,y2d,'LineWidth',3);
h=legend('a2','b2','c2','d2');
set(h,'FontSize',30);
set(gca,'FontSize',30);

figure

plot(x,yeval1,'LineWidth',3,x,yeval2,'LineWidth',3,x,yeval3,'LineWidth',3,x,yu1,'LineWidth',3,x,yu2,'LineWidth',3,x,yu3,'LineWidth',3);
h=legend('ANN approx of u1','ANN approx of u2','ANN approx of u3','u1','u2','u3');
set(h,'FontSize',30);
set(h,'Location','northeast');
set(gca,'FontSize',30);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% maximum and minimum error

max_diff=0;
min_diff=1000;
for i=1:ndataset
  n=dataset_in(i);
  eval1 = c1*1.0/(1.0+exp(-(a1*n+b1)))+d1;
  eval2 = c2*1.0/(1.0+exp(-(a2*n+b2)))+d2;
  eval3 = c3*1.0/(1.0+exp(-(a3*n+b3)))+d3;
  max_diff = max(max_diff,abs(eval1-dataset_u1(i)));
  max_diff = max(max_diff,abs(eval2-dataset_u2(i)));
  max_diff = max(max_diff,abs(eval3-dataset_u3(i)));
  min_diff = min(min_diff,abs(eval1-dataset_u1(i)));
  min_diff = min(min_diff,abs(eval2-dataset_u2(i)));
  min_diff = min(min_diff,abs(eval3-dataset_u3(i)));
end
max_diff
min_diff

end