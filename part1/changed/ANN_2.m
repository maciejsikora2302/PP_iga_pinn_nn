function ANN_2(layers_number)

%Preparation of dataset

%collection of dataset
A = [1/5 1/10 1/30; 1/10 2/15 1/10; 1/30 1/10 1/5];
i=1;
for n=0.01:0.01:0.5
    for x=0.01:0.001:0.5
    rhs= [ (pi*pi*n*n+2*cos(pi*n)-2)/(pi*pi*pi*n*n*n);
    (-2*pi*n*sin(pi*n)-4*cos(pi*n)+4)/(pi*pi*pi*n*n*n);
    ((2-pi*pi*n*n)*cos(pi*n)+2*pi*n*sin(pi*n)-2)/(pi*pi*pi*n*n*n) ];
    u=A \ rhs;
    y=u(1)*(1-x)^2+u(2)*2*x.*(1-x)+u(3)*x^2;
    dataset_in_n(i)=n;
    dataset_in_x(i)=x;
    dataset_y(i)=y;
    i=i+1;
    end
end
ndataset=i-1;

r = 0 + (1-0).*rand(ndataset,1);
r=r.*ndataset;
% Training

vec_a1=ones(1, layers_number);
vec_a2=ones(1, layers_number);
vec_b=ones(1, layers_number);
vec_c=zeros(1, layers_number);
vec_d=zeros(1, layers_number);

eta=0.1;

for idx=1:layers_number
    %Symbolic functions

    %Symbolic sigmoid
    syms z n e f g h p
    a1=sym('a1', [1, idx]);
    a2=sym('a2', [1, idx]);
    b=sym('b', [1, idx]);
    c=sym('c', [1, idx]);
    d=sym('d', [1, idx]);

    sigmoid(z,n,e,f,g,h,p) = h/(1+exp(-z*e-n*f-g))+p;
    
    result=sigmoid(z,n, a1(idx), a2(idx), b(idx), c(idx), d(idx));
    if(idx>1)
        for l=1:idx-1
            result=sigmoid(result,n,a1(idx-l),a2(idx-l),b(idx-l),c(idx-l),d(idx-l));
        end
    end
    ann_2(z,n,a1,a2,b,c,d)=result;
    
    y=(pi*pi*n*n+2*cos(pi*n)-2)/(pi*pi*pi*n*n*n)*(1-z)^2 + (-2*pi*n*sin(pi*n)-4*cos(pi*n)+4)/(pi*pi*pi*n*n*n)*(1-z) + ((2-pi*pi*n*n)*cos(pi*n)+2*pi*n*sin(pi*n)-2)/(pi*pi*pi*n*n*n)*z^2;
    y(z,n)=y;
    %Symbolic MSE
    temp=[a1, a2, b, c, d];
    combined=temp(:);
    combined=num2cell(combined);
    err(z,n,a1,a2,b,c,d)=0.5*(ann_2(z,n,combined{:}) - y(z,n))^2;
    
    for j=1:ndataset
      i=floor(r(j));

      if(i==0)
        i=1;
      end

     params(1:idx)=vec_a1(1:idx);
     params(idx+1:2*idx)=vec_a2(1:idx);
     params(2*idx+1:3*idx)=vec_b(1:idx);
     params(3*idx+1:4*idx)=vec_c(1:idx);
     params(4*idx+1:5*idx)=vec_d(1:idx);

     params_combined=num2cell(params);

     %Approximation of y
     eval = double(ann_2(dataset_in_x(i), dataset_in_n(i), params_combined{:}));
     error = 0.5*(eval-dataset_y(i))^2;

     %Symbolic differentation
     d_error_a1(z, n, a1, a2, b, c, d)=diff(err, a1(idx));
     d_error_a2(z,n,a1,a2, b, c, d)=diff(err, a2(idx));
     d_error_b(z,n,a1,a2, b, c, d)=diff(err, b(idx));
     d_error_c(z,n,a1,a2, b, c, d)=diff(err, c(idx));
     d_error_d(z,n,a1,a2, b, c, d)=diff(err, d(idx));

     derror_a1=double(d_error_a1(dataset_in_x(i),dataset_in_n(i), params_combined{:}));
     derror_a2=double(d_error_a2(dataset_in_x(i),dataset_in_n(i), params_combined{:}));
     derror_b=double(d_error_b(dataset_in_x(i),dataset_in_n(i), params_combined{:}));
     derror_c=double(d_error_c(dataset_in_x(i),dataset_in_n(i), params_combined{:}));
     derror_d=double(d_error_d(dataset_in_x(i),dataset_in_n(i), params_combined{:}));

     vec_a1(idx)=vec_a1(idx) - eta*derror_a1;
     vec_a2(idx)=vec_a2(idx) - eta*derror_a2;
     vec_b(idx)=vec_b(idx) - eta*derror_b;
     vec_c(idx)=vec_c(idx) - eta*derror_c;
     vec_d(idx)=vec_d(idx) - eta*derror_d;

     yeval(j+(idx*ndataset-ndataset))=eval;
     yy(j+(idx*ndataset-ndataset))=dataset_y(i);
     ya1(j+(idx*ndataset-ndataset))=a1(idx);
     ya2(j+(idx*ndataset-ndataset))=a2(idx);
     yb(j+(idx*ndataset-ndataset))=b(idx);
     yc(j+(idx*ndataset-ndataset))=c(idx);
     yd(j+(idx*ndataset-ndataset))=d(idx);
     ye(j+(idx*ndataset-ndataset))=error
     j
      
    end
    idx 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot convergence
x=1:ndataset*layers_number;

plot(x,ya1,x,ya2,x,yb,x,yc,x,yd);
h=legend('a1','a2','b','c','d');
set(h,'FontSize',20);


figure

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot error
x=1:layers_number*ndataset;
% hold on

plot(x,ye);
h=legend('error');
set(h,'FontSize',20);
set(gca, 'YScale', 'log');

end