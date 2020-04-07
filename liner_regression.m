function [ output_args_1,output_args_2,output_args_3 ] = liner_regression( input_X,Y,alpha,epoch,detla )
X=[ones(1,size(input_X,2));input_X];%x0-x(m+1),(m+1)*n
num_sample=size(X,2);
% gradient descending process
% initial values of parameters
THETA=mean(Y)*rand(1,size(X,1));
%learning rate
% if alpha is too large, the final error will be much large.
% if alpha is too small, the convergence will be slow
% epoch=500;
for k=1:epoch
    v_k=k;
    alpha_change=k/exp(k)+alpha;
    h_theta_x=THETA*X; % hypothesis function
    Jcost(k)=sum((h_theta_x-Y).^2)/num_sample;
    temp_delta=h_theta_x-Y;
    R=temp_delta*X';
    THETA=THETA-alpha_change*R/num_sample;
%     THETA=THETA-alpha*R/num_sample;
end
%% 通过代价函数的取（detla）值来控制程序结束
% k=1;
% while 1
%     v_k=k;
%     alpha_change=2*k/exp(k)+alpha;
%     h_theta_x=THETA*X; % hypothesis function
%     Jcost(k)=sum((h_theta_x-Y).^2)/num_sample;
%     temp_delta=h_theta_x-Y;
%     R=temp_delta*X';
%     THETA=THETA-alpha_change*R/num_sample;
% %     THETA=THETA-alpha*R/num_sample;
%     if Jcost(k)<=detla
%         break;
%     end
%     k=k+1;
% end
%% 
plot(Jcost)
if size(X,1)>2
    title('The iteration results of Multivariate Linear Regression model')
else 
    title('The iteration results of Univariate Linear Regression model')
end
xlabel('Iteration epoch')
ylabel('Cost function')
grid on
output_args_1=THETA;   
output_args_2=Jcost;
output_args_3=k;
end

