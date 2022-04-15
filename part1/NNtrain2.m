function NNtrain2()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Preparation of dataset

%collection of dataset
i=1;
for n=0.01:0.01:0.5
for x=0.01:0.01:0.5
y=sin(n*pi*x);
dataset_in_n(i)=n; 
dataset_in_x(i)=x; 
dataset_y(i)=y; 
i=i+1;
endfor
endfor
ndataset=i-1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%training
a1=1.0; a2=1.0; b=1.0; c=10.0; d=1.0;

eta=0.1;

r = 0 + (1-0).*rand(ndataset,1);
r=r.*ndataset;
for j=1:ndataset
  i=floor(r(j));
  if(i==0)
    i=1;
  endif
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %1
  a1_=a1; a2_=a2; b_=b; c_=c; d_=d;
  %evaluation of the first NN modeling a1
  eval = c*1.0/(1.0+exp(-(a1*dataset_in_n(i)+a2*dataset_in_x(i)+b)))+d;
  error = 0.5*(eval-dataset_y(i))^2;
  %a1=a1-eta*derivative of error with respect to a1
  %d error/da1 = \frac{cn exp \left(-a_1n-a_2x-b\right)\left(\frac{c}{exp\left(-a_1n-a_2x-b\right)+1}+d-y\right)}{\left(exp\left(-a_1n-a_2x-b\right)+1\right)^2}
  derrorda1 = ( c*dataset_in_n(i)*exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)*
                (c / (exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)+1)+d-dataset_y(i))
              ) / power((exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)+1),2);
  a1=a1-eta*  derrorda1;
  %a2=a1-eta*derivative of error with respect to a2
  %d error/da2 = \frac{cx exp \left(-a_1n-a_2x-b\right)\left(\frac{c}{exp\left(-a_1n-a_2x-b\right)+1}+d-y\right)}{\left(exp\left(-a_1n-a_2x-b\right)+1\right)^2}
  derrorda2 = ( c*dataset_in_x(i)*exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)*
                (c / (exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)+1)+d-dataset_y(i))
              ) / power((exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)+1),2);
  a2=a2-eta*  derrorda2;
  %b=b-eta*derivative of error with respect to b
  %d error/db = \frac{c exp \left(-a_1n-a_2x-b\right)\left(\frac{c}{exp\left(-a_1n-a_2x-b\right)+1}+d-y\right)}{\left(exp\left(-a_1n-a_2x-b\right)+1\right)^2}
  derrordb = ( c*exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)*
                (c / (exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)+1)+d-dataset_y(i))
              ) / power((exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)+1),2);
  b=b-eta*  derrordb;
  %c=c-eta*derivative of error with respect to c
  %d error/dc = \frac{\frac{c}{exp\left(-a_1n-a_2x-b\right)+1}+d-y}{exp\left(-a_1n-a_2x-b\right)+1} \\
  derrordc = ( c / (exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)+1)+d-dataset_y(i)
             ) / (exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)+1);
  c=c-eta*  derrordc;
  %d=d-eta*derivative of error with respect to d
  %d error/dd = \frac{c}{exp\left(-a_1n-a_2x-b\right)+1}+d-y
  derrordd = c / (exp(-a1*dataset_in_n(i)-a2*dataset_in_x(i)-b)+1)+d-dataset_y(i);
  d=d-eta*  derrordd;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%store for making plots later
  ya1(j)=a1;
  ya2(j)=a2;
  yb(j)=b;
  yc(j)=c;
  yd(j)=d;
  yeval(j)=eval;
  yy(j)=dataset_y(i);
  ye(j)=error;
%
endfor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot convergence
x=1:1:ndataset;
hold on

plot(x,ya1,x,ya2,x,yb,x,yc,x,yd);
h=legend('a1','a2','b','c','d');
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
% test single sin(n*pi*x)

figure
%test n
n=0.333;
%computing B-spline values
x=0:0.01:0.5;
y=sin(n*pi.*x);
eval = c*1.0./(1.0+exp(-(a1*n+a2.*x+b)))+d;
plot(x,y,x,eval);
h=legend('sin(n*pi*x)','ANN');
set(h,'FontSize',20);
set(h,'Location','northwest');
set(gca,'FontSize',20);


function y=sigma(x)
  y=1.0/(1.0+exp(-x));
  return
end

end