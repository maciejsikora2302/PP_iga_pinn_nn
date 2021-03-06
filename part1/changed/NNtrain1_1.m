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

vec_a1=ones(1, layers_number);
vec_b1=ones(1, layers_number);
vec_c1=ones(1, layers_number);
vec_d1=zeros(1, layers_number);

vec_a2=ones(1, layers_number);
vec_b2=ones(1, layers_number);
vec_c2=ones(1, layers_number);
vec_d2=zeros(1, layers_number);

vec_a3=ones(1, layers_number);
vec_b3=ones(1, layers_number);
vec_c3=ones(1, layers_number);
vec_d3=zeros(1, layers_number);

smallest_err_u1=1000;
smallest_err_u2=1000;
smallest_err_u3=1000;


biggest_err_u1=0;
biggest_err_u2=0;
biggest_err_u3=0;

eta1=0.1;
eta2=0.1;
eta3=0.1;
r = 0 + (1-0).*rand(ndataset,1);
r=r.*ndataset;
n=0.444;
for idx=1:layers_number
    %Symbolic functions

    %Symbolic sigmoid
    syms z a1 b1
    a=sym('a', [1, idx]);
    b=sym('b', [1, idx]);
    c=sym('c', [1, idx]);
    d=sym('d', [1, idx]);

    sigmoid(z,a1,b1) = a1/(1+exp(-z))+b1;
    
    result=sigmoid(z*a(idx)+b(idx), c(idx), d(idx));
    if(idx>1)
        for l=1:idx-1
            result=sigmoid(a(idx-l)*result+b(idx-l), c(idx-l), d(idx-l));
        end
    end
    ann_3(z,a,b,c,d)=result;


    %Symbolic first and second derivative of PINN with respect to x
    
    %Symbolic PDE solver function
    temp=[a, b, c, d];
    combined=temp(:);
    combined=num2cell(combined);

    %Symbolic error1 function
    error1(z, a, b, c, d)=0.5 * (  ...
      ann_3(z,combined{:})  ...
      - ( ...
        ((pi*pi*n*n+2*cos(pi*n)-2)/(pi*pi*pi*n*n*n)) * (1-z)^2 ...
        ) ...
      )^2;

    error2(z, a, b, c, d)=0.5 * ( ...
      ann_3(z,combined{:})  ...
      - ( ...
        ((-2*pi*n*sin(pi*n)-4*cos(pi*n)+4)/(pi*pi*pi*n*n*n)) * 2*z*(1-z) ...
        ) ...
      )^2;

    error3(z, a, b, c, d)=0.5 * ( ...
      ann_3(z,combined{:})  ...
      - ( ...
        (((2-pi*pi*n*n)*cos(pi*n)+2*pi*n*sin(pi*n)-2)/(pi*pi*pi*n*n*n)) * z^2 ...
        ) ...
      )^2;
    

  for j=1:ndataset
    params_u1(1:idx)=vec_a1(1:idx);
    params_u1(idx+1:2*idx)=vec_b1(1:idx);
    params_u1(2*idx+1:3*idx)=vec_c1(1:idx);
    params_u1(3*idx+1:4*idx)=vec_d1(1:idx);
    
    params_u2(1:idx)=vec_a2(1:idx);
    params_u2(idx+1:2*idx)=vec_b2(1:idx);
    params_u2(2*idx+1:3*idx)=vec_c2(1:idx);
    params_u2(3*idx+1:4*idx)=vec_d2(1:idx);

    params_u3(1:idx)=vec_a3(1:idx);
    params_u3(idx+1:2*idx)=vec_b3(1:idx);
    params_u3(2*idx+1:3*idx)=vec_c3(1:idx);
    params_u3(3*idx+1:4*idx)=vec_d3(1:idx);

    params_u1_combined=num2cell(params_u1);
    params_u2_combined=num2cell(params_u2);
    params_u3_combined=num2cell(params_u3);
    i=floor(r(j));
    if(i==0)
      i=1;
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Approximation of u1, u2 and u3 coefficients
    eval_u1 = double(ann_3(dataset_in(i), params_u1_combined{:}));
    eval_u2 = double(ann_3(dataset_in(i), params_u2_combined{:}));
    eval_u3 = double(ann_3(dataset_in(i), params_u3_combined{:}));

    %Errors
    error_u1 = 0.5*(eval_u1-dataset_u1(i))^2;
    error_u2 = 0.5*(eval_u2-dataset_u2(i))^2;
    error_u3 = 0.5*(eval_u3-dataset_u3(i))^2;
    %Symbolic differentation
    d_error_1_a(z, a, b, c, d)=diff(error1, a(idx));
    d_error_1_b(z, a, b, c, d)=diff(error1, b(idx));
    d_error_1_c(z, a, b, c, d)=diff(error1, c(idx));
    d_error_1_d(z, a, b, c, d)=diff(error1, d(idx));
    
    derror_u1_1a=double(d_error_1_a(dataset_in(i), params_u1_combined{:}));
    derror_u1_1b=double(d_error_1_b(dataset_in(i), params_u1_combined{:}));
    derror_u1_1c=double(d_error_1_c(dataset_in(i), params_u1_combined{:}));
    derror_u1_1d=double(d_error_1_d(dataset_in(i), params_u1_combined{:}));

    derror_u2_1a=double(d_error_1_a(dataset_in(i), params_u2_combined{:}));
    derror_u2_1b=double(d_error_1_b(dataset_in(i), params_u2_combined{:}));
    derror_u2_1c=double(d_error_1_c(dataset_in(i), params_u2_combined{:}));
    derror_u2_1d=double(d_error_1_d(dataset_in(i), params_u2_combined{:}));

    derror_u3_1a=double(d_error_1_a(dataset_in(i), params_u3_combined{:}));
    derror_u3_1b=double(d_error_1_b(dataset_in(i), params_u3_combined{:}));
    derror_u3_1c=double(d_error_1_c(dataset_in(i), params_u3_combined{:}));
    derror_u3_1d=double(d_error_1_d(dataset_in(i), params_u3_combined{:}));
    
    % Training of the boundary condition at x=0
    d_error_2_a(z, a, b, c, d)=diff(error2, a(idx));
    d_error_2_b(z, a, b, c, d)=diff(error2, b(idx));
    d_error_2_c(z, a, b, c, d)=diff(error2, c(idx));
    d_error_2_d(z, a, b, c, d)=diff(error2, d(idx));

    derror_u1_2a=double(d_error_2_a(dataset_in(i), params_u1_combined{:}));
    derror_u1_2b=double(d_error_2_b(dataset_in(i), params_u1_combined{:}));
    derror_u1_2c=double(d_error_2_c(dataset_in(i), params_u1_combined{:}));
    derror_u1_2d=double(d_error_2_d(dataset_in(i), params_u1_combined{:}));

    derror_u2_2a=double(d_error_2_a(dataset_in(i), params_u2_combined{:}));
    derror_u2_2b=double(d_error_2_b(dataset_in(i), params_u2_combined{:}));
    derror_u2_2c=double(d_error_2_c(dataset_in(i), params_u2_combined{:}));
    derror_u2_2d=double(d_error_2_d(dataset_in(i), params_u2_combined{:}));

    derror_u3_2a=double(d_error_2_a(dataset_in(i), params_u3_combined{:}));
    derror_u3_2b=double(d_error_2_b(dataset_in(i), params_u3_combined{:}));
    derror_u3_2c=double(d_error_2_c(dataset_in(i), params_u3_combined{:}));
    derror_u3_2d=double(d_error_2_d(dataset_in(i), params_u3_combined{:}));
    
    
    % Training of the boundary condition at x=0.5
    d_error_3_a(z, a ,b, c, d)=diff(error3, a(idx));
    d_error_3_b(z, a, b, c, d)=diff(error3, b(idx));
    d_error_3_c(z, a, b, c, d)=diff(error3, c(idx));
    d_error_3_d(z, a, b, c, d)=diff(error3, d(idx));

    derror_u1_3a=double(d_error_3_a(dataset_in(i), params_u1_combined{:}));
    derror_u1_3b=double(d_error_3_b(dataset_in(i), params_u1_combined{:}));
    derror_u1_3c=double(d_error_3_c(dataset_in(i), params_u1_combined{:}));
    derror_u1_3d=double(d_error_3_d(dataset_in(i), params_u1_combined{:}));

    derror_u2_3a=double(d_error_3_a(dataset_in(i), params_u2_combined{:}));
    derror_u2_3b=double(d_error_3_b(dataset_in(i), params_u2_combined{:}));
    derror_u2_3c=double(d_error_3_c(dataset_in(i), params_u2_combined{:}));
    derror_u2_3d=double(d_error_3_d(dataset_in(i), params_u2_combined{:}));

    derror_u3_3a=double(d_error_3_a(dataset_in(i), params_u3_combined{:}));
    derror_u3_3b=double(d_error_3_b(dataset_in(i), params_u3_combined{:}));
    derror_u3_3c=double(d_error_3_c(dataset_in(i), params_u3_combined{:}));
    derror_u3_3d=double(d_error_3_d(dataset_in(i), params_u3_combined{:}));
    
    vec_a1(idx)=vec_a1(idx) - eta1*(derror_u1_1a);
    vec_b1(idx)=vec_b1(idx) - eta1*(derror_u1_1b);
    vec_c1(idx)=vec_c1(idx) - eta1*(derror_u1_1c);
    vec_d1(idx)=vec_d1(idx) - eta1*(derror_u1_1d);

    vec_a2(idx)=vec_a2(idx) - eta2*(derror_u2_2a);
    vec_b2(idx)=vec_b2(idx) - eta2*(derror_u2_2b);
    vec_c2(idx)=vec_c2(idx) - eta2*(derror_u2_2c);
    vec_d2(idx)=vec_d2(idx) - eta2*(derror_u2_2d);

    vec_a3(idx)=vec_a3(idx) - eta3*(derror_u3_3a);
    vec_b3(idx)=vec_b3(idx) - eta3*(derror_u3_3b);
    vec_c3(idx)=vec_c3(idx) - eta3*(derror_u3_3c);
    vec_d3(idx)=vec_d3(idx) - eta3*(derror_u3_3d);

    if(error_u1 < smallest_err_u1)
            smallest_err_u1=error_u1;
            best_params_u1=num2cell(params_u1);
            min_diff_u1=error_u1;
        end

        if(error_u1 > biggest_err_u1)
            biggest_err_u1=error_u1;
            max_diff_u1=error_u1;
        end

        if(error_u2 < smallest_err_u2)
            smallest_err_u2=error_u2;
            best_params_u2=num2cell(params_u2);
            min_diff_u2=error_u2;
        end

        if(error_u2 > biggest_err_u2)
            biggest_err_u2=error_u2;
            max_diff_u2=error_u2;
        end

        if(error_u3 < smallest_err_u3)
            smallest_err_u3=error_u3;
            best_params_u3=num2cell(params_u3);
            min_diff_u3=error_u3;
        end

        if(error_u3 > biggest_err_u3)
            biggest_err_u3=error_u3;
            max_diff_u3=error_u3;
        end

    yeval_u1(j+(idx*ndataset-ndataset))=eval_u1;
    yeval_u2(j+(idx*ndataset-ndataset))=eval_u2;
    yeval_u3(j+(idx*ndataset-ndataset))=eval_u3;

    ya_u1(j+(idx*ndataset-ndataset))=vec_a1(idx);
    yb_u1(j+(idx*ndataset-ndataset))=vec_b1(idx);
    yc_u1(j+(idx*ndataset-ndataset))=vec_c1(idx);
    yd_u1(j+(idx*ndataset-ndataset))=vec_d1(idx);

    ya_u2(j+(idx*ndataset-ndataset))=vec_a2(idx);
    yb_u2(j+(idx*ndataset-ndataset))=vec_b2(idx);
    yc_u2(j+(idx*ndataset-ndataset))=vec_c2(idx);
    yd_u2(j+(idx*ndataset-ndataset))=vec_d2(idx);

    ya_u3(j+(idx*ndataset-ndataset))=vec_a3(idx);
    yb_u3(j+(idx*ndataset-ndataset))=vec_b3(idx);
    yc_u3(j+(idx*ndataset-ndataset))=vec_c3(idx);
    yd_u3(j+(idx*ndataset-ndataset))=vec_d3(idx);


    yy_u1(j+(idx*ndataset-ndataset))=dataset_u1(i);
    yy_u2(j+(idx*ndataset-ndataset))=dataset_u2(i);
    yy_u3(j+(idx*ndataset-ndataset))=dataset_u3(i);
    ye_u1(j+(idx*ndataset-ndataset))=error_u1
    ye_u2(j+(idx*ndataset-ndataset))=error_u2;
    ye_u3(j+(idx*ndataset-ndataset))=error_u3;
    j
  end
  idx
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot convergence
% plot convergence
x=1:layers_number*ndataset;
% hold on
% Vectors parameters  have parameters from every layer on the same plot!
plot(x,ya_u1,x,yb_u1,x,yc_u1,x,yd_u1);
h=legend('a_{u1}', 'b_{u1}', 'c_{u1}', 'd_{u1}');
set(h,'FontSize',10);
figure

plot(x,ya_u2,x,yb_u2,x,yc_u2,x,yd_u2);
h=legend('a_{u2}', 'b_{u2}', 'c_{u2}', 'd_{u2}');
set(h,'FontSize',10);
figure

plot(x,ya_u3,x,yb_u3,x,yc_u3,x,yd_u3);
h=legend('a_{u3}', 'b_{u3}', 'c_{u3}', 'd_{u3}');
set(h,'FontSize',10);

figure

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% plot error
x=1:1:layers_number*ndataset;
% hold on

plot(x,ye_u1,x,ye_u2,x,ye_u3);
h=legend('error(u1)', 'error(u2)', 'error(u3)');
set(h,'FontSize',20);
set(gca, 'YScale', 'log');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing maximum and minimum difference for u(1), u(2) and u(3)
max_diff_u1
min_diff_u1

max_diff_u2
min_diff_u2

max_diff_u3
min_diff_u3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
best_params_u1
best_params_u2
best_params_u3

end