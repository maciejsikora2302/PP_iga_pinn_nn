function data()
    A = [1/5 1/10 1/30; 1/10 2/15 1/10; 1/30 1/10 1/5];
    i=1;
    %for n=0.01:0.0000004:0.5
    start = 0.01;
    end_range = 20.0;
    quantity = 20000;
    for n=start:(end_range-start)/quantity:end_range
    rhs= [ (pi*pi*n*n+2*cos(pi*n)-2)/(pi*pi*pi*n*n*n);
    (-2*pi*n*sin(pi*n)-4*cos(pi*n)+4)/(pi*pi*pi*n*n*n);
    ((2-pi*pi*n*n)*cos(pi*n)+2*pi*n*sin(pi*n)-2)/(pi*pi*pi*n*n*n) ];
    u=A \ rhs;
    in(i)=n;
    u1(i)=u(1);
    u2(i)=u(2);
    u3(i)=u(3);
    i=i+1;
    end
    T = table(in, u1, u2, u3);
    writetable(T, 'dataset20k_to20.txt')
end