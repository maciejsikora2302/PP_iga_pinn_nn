function NNtrain3()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Preparation of dataset

%collection of dataset
i=1;
n=0.444;
for x=0.01:0.01:0.5
y=sin(n*pi*x);
dataset_in_n(i)=n; 
dataset_in_x(i)=x; 
dataset_y(i)=y; 
i=i+1;
endfor
ndataset=i-1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
a=1.0; b=1.0; c=3.0; d=1.0;

eta=0.1;
eta2=0.1;
eta3=0.1;

r = 0 + (1-0).*rand(ndataset,1);
r=r.*ndataset;
for j=1:ndataset
  i=floor(r(j));
  if(i==0)
    i=1;
  endif
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %evaluation of the first NN modeling a1
  eval = c*1.0/(1.0+exp(-(a*dataset_in_x(i)+b)))+d;
  error = 0.5*(eval-dataset_y(i))^2;
  x = dataset_in_x(i);
  n = dataset_in_n(i);
  Fx = c*(2*a*a*exp(-2*a*x-2*b) / power((exp(-a*x-b)+1),3) - a*a*exp(-a*x-b) / power((exp(-a*x-b)+1),2))+n*n*pi*pi*sin(n*x);
  derror1da = Fx*c *( (a*a*x*exp(-a*x-b))/power((exp(-a*x-b)+1),2)  -(6*a*a*x*exp(-2*a*x-2*b))/power((exp(-a*x-b)+1),3) +(6*a*a*x*exp(-3*a*x-3*b))/power((exp(-a*x-b)+1),4) -(2*a*exp(-a*x-b))/power((exp(-a*x-b)+1),2) +(4*a*exp(-2*a*x-2*b))/power((exp(-a*x-b)+1),3) );
  a=a-eta*  derror1da;
  derror1db =  Fx*c*( (a*a*exp(a*x+b))*(-4*exp(a*x+b)+exp(2*a*x+2*b)+1)/power((exp(a*x+b)+1),4));
  b=b-eta*  derror1db;
  derror1dc = Fx*( (2*a*a*exp(-2*a*x-2*b))/power((exp(-a*x-b)+1),3)-(a*a*exp(-a*x-b))/power((exp(-a*x-b)+1),2));
  c=c-eta*  derror1dc;
  derror1dd = 0;
  d=d-eta*  derror1dd;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ya(j)=a;
  yb(j)=b;
  yc(j)=c;
  yd(j)=d;
  yeval(j)=eval;
  yy(j)=dataset_y(i);
  ye(j)=error;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%b.c. 
for i=1:1
  %b.c. at 0
  x=0;
  derror2da = 0;
  a=a-eta2*  derror2da;
  derror2db = (c / (1+exp(-b))+d)* (exp(-b)*c)/power((exp(-b)+1),2);
  b=b-eta2*  derror2db;
  derror2dc = (c/(1+exp(-b))+d)*( (2*a*a*exp(-2*a*x-2*b))/power(exp(-a*x-b)+1,3) - (a*a*exp(-a*x-b))/power(exp(-a*x-b)+1,2) );
  c=c-eta2*  derror2dc;
  derror2dd = c/(1+exp(-b))+d;  
  d=d-eta2*  derror2dd;
  x=0.5;
  derror3da = (a*c*exp(-a*0.5-b)/power((exp(-a*0.5-b)+1),2)-n*pi*cos(n*pi*0.5)) *c*exp(b-a)*((1-0.5*a)*exp(2*a+b)+(0.5*a+1)*exp(1.5*a))/power((exp(0.5*a+b)+1),3);
  a=a-eta3*  derror3da;
  derror3db = (a*c*exp(-a*0.5-b)/power((exp(-a*0.5-b)+1),2)-n*pi*cos(n*pi*0.5))* a*c*exp(b-0.5*a)*(exp(a)+exp(1.5*a+b))/power((exp(0.5*a+b)+1),3);
  b=b-eta3*  derror3db;
  derror3dc = (a*c*exp(-a*0.5-b)/power((exp(-a*0.5-b)+1),2)-n*pi*cos(n*pi*0.5))* a*exp(-b-0.5*a)/power((exp(-0.5*a-b)+1),2);
  c=c-eta3*  derror3dc;
  derror3dd = 0;
  d=d-eta3*  derror3dd;
endfor
  
%
endfor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot convergence
x=1:1:ndataset;
hold on

plot(x,ya,x,yb,x,yc,x,yd);
h=legend('a','b','c','d');
set(h,'FontSize',20);

figure

plot(x,yeval,x,yy);
h=legend('ANN approx of y','y');
set(h,'FontSize',20);
set(h,'Location','northeast');
set(gca,'FontSize',20);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot error
x=1:1:ndataset;
hold on

figure

plot(x,ye);
h=legend('error');
set(h,'FontSize',20);
set(gca, 'YScale', 'log');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test single sin(0.333*pi*x)

figure
%test n
n=0.444;
%computing B-spline values
x=0:0.01:0.5;
y=sin(n*pi.*x);
eval = c./(1.0+exp(-a.*x-b))+d;
plot(x,y,x,eval);
h=legend('sin(0.444*pi*x)','ANN');
set(h,'FontSize',20);
set(h,'Location','northwest');
set(gca,'FontSize',20);



function y=sigma(x)
  y=1.0/(1.0+exp(-x));
  return
end

end