function ANN_3(layers_number)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Creation of dataset
% Here we solve the projection of the known solution u=sin(n*pi*x)
A = [1/5 1/10 1/30; 1/10 2/15 1/10; 1/30 1/10 1/5];
i=1;
for n=0.01:0.01:0.5
rhs= [ (pi*pi*n*n+2*cos(pi*n)-2)/(pi*pi*pi*n*n*n);
(-2*pi*n*sin(pi*n)-4*cos(pi*n)+4)/(pi*pi*pi*n*n*n);
((2-pi*pi*n*n)*cos(pi*n)+2*pi*n*sin(pi*n)-2)/(pi*pi*pi*n*n*n) ];
u=A \ rhs;
dataset_in(i)=n;
dataset_u1(i)=u(1);
dataset_u2(i)=u(2);
dataset_u3(i)=u(3);
i=i+1;
end
ndataset=i-1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
%Initialization for every (three in this case) coefficients of a linear
%combination of B-spline functions
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
    ann_3(z,a,b,c,d)=result


    %Symbolic first and second derivative of PINN with respect to x
    ann_3_x(z,a,b,c,d)=diff(ann_3, z);
    ann_3_xx(z,a,b,c,d)=diff(ann_3_x, z);
    
    %Symbolic PDE solver function
    temp=[a, b, c, d];
    combined=temp(:);
    combined=num2cell(combined);
    F(z,a, b, c, d)=ann_3_xx(z, combined{:}) + n^2*pi^2*sin(n*pi*z);
    
    %Symbolic error1 function
    error1(z, a, b, c, d)=0.5*F(z, combined{:})^2;
    
    %Symbolic error2 function
    error2(a, b, c, d)=0.5*ann_3(0, combined{:});
    
    %Symbolic error3 function
    error3(a, b, c, d)=0.5*(ann_3_x(0.5, combined{:}) - n*pi*cos(n*pi*0.5))^2;

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
        d_error_1_b(z,a, b, c, d)=diff(error1, b(idx));
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
        d_error_2_a(a, b, c, d)=diff(error2, a(idx));
        d_error_2_b(a, b, c, d)=diff(error2, b(idx));
        d_error_2_c(a, b, c, d)=diff(error2, c(idx));
        d_error_2_d(a, b, c, d)=diff(error2, d(idx));

        derror_u1_2a=double(d_error_2_a(params_u1_combined{:}));
        derror_u1_2b=double(d_error_2_b(params_u1_combined{:}));
        derror_u1_2c=double(d_error_2_c(params_u1_combined{:}));
        derror_u1_2d=double(d_error_2_d(params_u1_combined{:}));

        derror_u2_2a=double(d_error_2_a(params_u2_combined{:}));
        derror_u2_2b=double(d_error_2_b(params_u2_combined{:}));
        derror_u2_2c=double(d_error_2_c(params_u2_combined{:}));
        derror_u2_2d=double(d_error_2_d(params_u2_combined{:}));

        derror_u3_2a=double(d_error_2_a(params_u3_combined{:}));
        derror_u3_2b=double(d_error_2_b(params_u3_combined{:}));
        derror_u3_2c=double(d_error_2_c(params_u3_combined{:}));
        derror_u3_2d=double(d_error_2_d(params_u3_combined{:}));
        
        
        % Training of the boundary condition at x=0.5
        d_error_3_a(a,b, c, d)=diff(error3, a(idx));
        d_error_3_b(a, b, c, d)=diff(error3, b(idx));
        d_error_3_c(a,b, c, d)=diff(error3, c(idx));
        d_error_3_d(a, b, c, d)=diff(error3, d(idx));

        derror_u1_3a=double(d_error_3_a(params_u1_combined{:}));
        derror_u1_3b=double(d_error_3_b(params_u1_combined{:}));
        derror_u1_3c=double(d_error_3_c(params_u1_combined{:}));
        derror_u1_3d=double(d_error_3_d((params_u1_combined{:})));

        derror_u2_3a=double(d_error_3_a(params_u2_combined{:}));
        derror_u2_3b=double(d_error_3_b(params_u2_combined{:}));
        derror_u2_3c=double(d_error_3_c(params_u2_combined{:}));
        derror_u2_3d=double(d_error_3_d(params_u2_combined{:}));

        derror_u3_3a=double(d_error_3_a(params_u3_combined{:}));
        derror_u3_3b=double(d_error_3_b(params_u3_combined{:}));
        derror_u3_3c=double(d_error_3_c(params_u3_combined{:}));
        derror_u3_3d=double(d_error_3_d(params_u3_combined{:}));
        
        vec_a1(idx)=vec_a1(idx) - eta1*(derror_u1_3a+derror_u1_2a+derror_u1_1a);
        vec_b1(idx)=vec_b1(idx) - eta1*(derror_u1_3b+derror_u1_2b+derror_u1_1b);
        vec_c1(idx)=vec_c1(idx) - eta1*(derror_u1_3c+derror_u1_2c+derror_u1_1c);
        vec_d1(idx)=vec_d1(idx) - eta1*(derror_u1_3d+derror_u1_2d+derror_u1_1d);

        vec_a2(idx)=vec_a2(idx) - eta2*(derror_u2_3a+derror_u2_2a+derror_u2_1a);
        vec_b2(idx)=vec_b2(idx) - eta2*(derror_u2_3b+derror_u2_2b+derror_u2_1b);
        vec_c2(idx)=vec_c2(idx) - eta2*(derror_u2_3c+derror_u2_2c+derror_u2_1c);
        vec_d2(idx)=vec_d2(idx) - eta2*(derror_u2_3d+derror_u2_2d+derror_u2_1d);

        vec_a3(idx)=vec_a3(idx) - eta3*(derror_u3_3a+derror_u3_2a+derror_u3_1a);
        vec_b3(idx)=vec_b3(idx) - eta3*(derror_u3_3b+derror_u3_2b+derror_u3_1b);
        vec_c3(idx)=vec_c3(idx) - eta3*(derror_u3_3c+derror_u3_2c+derror_u3_1c);
        vec_d3(idx)=vec_d3(idx) - eta3*(derror_u3_3d+derror_u3_2d+derror_u3_1d);

        if(error_u1 < smallest_err_u1)
            smallest_err_u1=error_u1;
            best_params_u1=num2cell(params_u1);
        end

        if(error_u2 < smallest_err_u2)
            smallest_err_u2=error_u2;
            best_params_u2=num2cell(params_u2);
        end

        if(error_u3 < smallest_err_u3)
            smallest_err_u3=error_u3;
            best_params_u3=num2cell(params_u3);
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
x=1:layers_number*ndataset;
% hold on
yeval_u1-yy_u1
plot(x,ya_u1,x,yb_u1,x,yc_u1,x,yd_u1,x,ya_u2,x,yb_u2,x,yc_u2,x,yd_u2,x,ya_u3,x,yb_u3,x,yc_u3,x,yd_u3);
h=legend('a_u1', 'b_u1', 'c_u1', 'd_u1','a_u2', 'b_u2', 'c_u2', 'd_u2','a_u3', 'b_u3', 'c_u3', 'd_u3');
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
% Computing maximum and minimum difference for u(1)
% max_diff_u1=0; 
% min_diff_u1=1000;
% 
% max_diff_u2=0; 
% min_diff_u2=1000;
% 
% max_diff_u3=0; 
% min_diff_u3=1000;
% 
% for i=1:ndataset*layers_number
%     k=mod(i, ndataset);
%     if(k==0)
%         k=1;
%     end
%     n=dataset_in(k);
%     max_diff_u1 = double(max(max_diff_u1, abs(dataset_u1(k)-ann_3(n, best_params_u1{:}))));
%     min_diff_u1 = double(min(min_diff_u1,abs(dataset_u1(k)-ann_3(n, best_params_u1{:}))));
% 
%     max_diff_u2 = double(max(max_diff_u2, abs(dataset_u2(k)-ann_3(n, best_params_u2{:}))));
%     min_diff_u2 = double(min(min_diff_u2,abs(dataset_u2(k)-ann_3(n, best_params_u2{:}))));
% 
%     max_diff_u3 = double(max(max_diff_u3, abs(dataset_u3(k)-ann_3(n, best_params_u3{:}))));
%     min_diff_u3 = double(min(min_diff_u3,abs(dataset_u3(k)-ann_3(n, best_params_u3{:}))));
% end
% max_diff_u1
% min_diff_u1
% 
% max_diff_u2
% min_diff_u2
% 
% max_diff_u3
% min_diff_u3
end