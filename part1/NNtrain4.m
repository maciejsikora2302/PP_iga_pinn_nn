function NNtrain3()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Preparation of dataset

%collection of dataset
i=1;
n=0.333;
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
  %a=a-eta*derivative of error with respect to a
  %d error/da = c\left(\frac{a^2x*exp(-ax-b)}{(exp(-ax-b)+1)^2}
%-\frac{6a^2x*exp(-2ax-2b)}{(exp(-ax-b)+1)^3}
%+\frac{6a^2x*exp(-3ax-3b)}{(exp(-ax-b)+1)^4}
%-\frac{2aexp(-ax-b)}{(exp(-ax-b)+1)^2}
%+\frac{4a*exp(-2ax-2b)}{(exp(-ax-b)+1)^3}\right)
  Fx = c*(2*a*a*exp(-2*a*x-2*b) / power((exp(-a*x-b)+1),3) - a*a*exp(-a*x-b) / power((exp(-a*x-b)+1),2))+n*n*pi*pi*sin(n*x);
  derror1da = Fx*c *( (a*a*x*exp(-a*x-b))/power((exp(-a*x-b)+1),2)  -(6*a*a*x*exp(-2*a*x-2*b))/power((exp(-a*x-b)+1),3) +(6*a*a*x*exp(-3*a*x-3*b))/power((exp(-a*x-b)+1),4) -(2*a*exp(-a*x-b))/power((exp(-a*x-b)+1),2) +(4*a*exp(-2*a*x-2*b))/power((exp(-a*x-b)+1),3) );
  a=a-eta*  derror1da;
  %b=b-eta*derivative of error with respect to b
  %d error/db = c\left(\frac{a^2exp(ax+b)\left(-4exp(ax+b)+exp(2ax+2b)+1\right)}{(exp(ax+b)+1)^4}\right)
  derror1db =  Fx*c*( (a*a*exp(a*x+b))*(-4*exp(a*x+b)+exp(2*a*x+2*b)+1)/power((exp(a*x+b)+1),4));
  b=b-eta*  derror1db;
  %c=c-eta*derivative of error with respect to c
  %d error/dc = \left(\frac{2a^2 exp(-2ax-2b)}{\left(exp(-ax-b)+1\right)^3}-\frac{a^2 exp(-ax-b)}{\left(exp(-ax-b)+1\right)^2}\right)
  derror1dc = Fx*( (2*a*a*exp(-2*a*x-2*b))/power((exp(-a*x-b)+1),3)-(a*a*exp(-a*x-b))/power((exp(-a*x-b)+1),2));
  c=c-eta*  derror1dc;
  %d=d-eta*derivative of error with respect to d
  %d error/dd = 0
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
  %b=b-eta*derivative of error with respect to b
  derror2db = (c / (1+exp(-b))+d)* (exp(-b)*c)/power((exp(-b)+1),2);
  b=b-eta2*  derror2db;
  %c=c-eta*derivative of error with respect to c
  derror2dc = (c/(1+exp(-b))+d)*( (2*a*a*exp(-2*a*x-2*b))/power(exp(-a*x-b)+1,3) - (a*a*exp(-a*x-b))/power(exp(-a*x-b)+1,2) );
  c=c-eta2*  derror2dc;
  %d=d-eta*derivative of error with respect to d
  derror2dd = c/(1+exp(-b))+d;  
  d=d-eta2*  derror2dd;
  % b.c. at 0.5
  x=0.5;
  derror3da = (a*c*exp(-a*0.5-b)/power((exp(-a*0.5-b)+1),2)-n*pi*cos(n*pi*0.5)) *c*exp(b-a)*((1-0.5*a)*exp(2*a+b)+(0.5*a+1)*exp(1.5*a))/power((exp(0.5*a+b)+1),3);
  a=a-eta3*  derror3da;
  %b=b-eta*derivative of error with respect to b
  derror3db = (a*c*exp(-a*0.5-b)/power((exp(-a*0.5-b)+1),2)-n*pi*cos(n*pi*0.5))* a*c*exp(b-0.5*a)*(exp(a)+exp(1.5*a+b))/power((exp(0.5*a+b)+1),3);
  b=b-eta3*  derror3db;
  %c=c-eta*derivative of error with respect to c
  derror3dc = (a*c*exp(-a*0.5-b)/power((exp(-a*0.5-b)+1),2)-n*pi*cos(n*pi*0.5))* a*exp(-b-0.5*a)/power((exp(-0.5*a-b)+1),2);
  c=c-eta3*  derror3dc;
  %d=d-eta*derivative of error with respect to d
  derror3dd = 0;
  d=d-eta3*  derror3dd;
endfor
  
%
endfor

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot convergence
x=1:1:ndataset;
hold on

plot(x,ya,'LineWidth',6,x,yb,'LineWidth',6,x,yc,'LineWidth',6,x,yd,'LineWidth',6);
h=legend('a','b','c','d');
set(h,'FontSize',20);

figure

plot(x,yeval,'LineWidth',6,x,yy,'LineWidth',6);
h=legend('ANN approx of y','y');
set(h,'FontSize',20);
set(h,'Location','northeast');
set(gca,'FontSize',20);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot error
x=1:1:ndataset;
hold on

figure

plot(x,ye,'LineWidth',6);
h=legend('error');
set(h,'FontSize',20);
set(gca, 'YScale', 'log');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test single sin(0.333*pi*x)

figure
%test n
n=0.333;
%computing B-spline values
x=0:0.01:0.5;
y=sin(n*pi.*x);
eval = c./(1.0+exp(-a.*x-b))+d;
plot(x,y,'LineWidth',6,x,eval,'LineWidth',6);
h=legend('sin(0.333*pi*x)','ANN');
set(h,'FontSize',40);
set(h,'Location','northwest');
set(gca,'FontSize',40);



function y=sigma(x)
  y=1.0/(1.0+exp(-x));
  return
end

end