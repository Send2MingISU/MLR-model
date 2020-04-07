clc
clear
load data 
load label
%% ���ݴ������쳣ֵ�޳� ��ӦT1 T2
% ���ӻ�
Example=data(:,11);
X1=data(:,1:18);
X2=data(:,19:36);
for i=1:size(X1,2)%�����ݽ��й�һ�����Ա��ڿ��ӻ�
    X1(:,i)=(X1(:,i)-min(X1(:,i)))/(max(X1(:,i))-min(X1(:,i)));
    X2(:,i)=(X2(:,i)-min(X2(:,i)))/(max(X2(:,i))-min(X2(:,i)));
end
figure(1);
subplot(2,1,1)
boxplot(X1,'notch','on','labels',{feat(1:18)'},'whisker',1)
% boxplot(X1,'notch','on','whisker',1)
subplot(2,1,2)
boxplot(X2,'notch','on','labels',{feat(19:36)'},'whisker',1)
% boxplot(X2,'notch','on','whisker',1)

% ���쳣���ݽ����޳�
temp_QL_QU=[];
for i=1:size(data,2)
    temp_QL_QU=[temp_QL_QU;quantile(data(:,i),[0.25 0.75]) ];%[L,U]
end
temp_IQR=temp_QL_QU(:,2)-temp_QL_QU(:,1);
Beta=1;%
Delete_QL=temp_QL_QU(:,1)-Beta*temp_IQR;
Delete_QU=temp_QL_QU(:,2)+Beta*temp_IQR;
Delete_loc=[];
Selection=[1:36];%�˹�ѡ��Ҫ����ı���
for i=Selection
    Delete_loc=[Delete_loc;find(data(:,i)>Delete_QU(i)|data(:,i)<Delete_QL(i))];
end
Delete_loc_final=unique(Delete_loc);
data(Delete_loc_final,:)=[];

%��ȱʧ���ݽ��в���,������λ����ȱʧ���ݽ��в�ȫ
for i=1:size(data,2)
    data(isnan(data(:,i)),i)=prctile(data(:,i),50);
end
% �����ͼ��
figure(2)
subplot(2,1,1)
plot(Example);
ylabel('Sea d.[m]')
xlabel('Points')
title('Data of Sea depth before processing')
subplot(2,1,2)
plot(data(:,11));
ylabel('Sea d.[m]')
xlabel('Poinst')
title('Data of Sea depth after processing')
%% ����pearson�����ϵ����������ѡ��
for i=1:36
    Pearson(i)=corr(data(:,i),data(:,35),'type','Pearson');
end
Judgement=abs(Pearson);
Feature_selection=find(Judgement>0.5);
bar(Pearson)
axis([0 37 -0.4 1.1])
xlabel('The number of features')
ylabel('Pearson Coefficient')
grid on
%% �������ݹ�һ��
for i=1:size(data,2)%�����ݽ��й�һ�����Ա��ڿ��ӻ�
    data(:,i)=(data(:,i)-min(data(:,i)))/(max(data(:,i))-min(data(:,i)));
end
%% ���������Իع�ģ��
%�ݶ��½���
X_Gradient_Descent=data(:,26)';
Y_Gradient_Descent=data(:,35)';
alpha=0.05;%��������Ӧ�ķ�����k/exp(k)+0.05
epoch=800;
detla=0.006;
[Theta_Gradient_Descent,Jcost,epoch]=liner_regression(X_Gradient_Descent,Y_Gradient_Descent,alpha,epoch,detla );
MES_GD=sum((X_Gradient_Descent*Theta_Gradient_Descent(2)+Theta_Gradient_Descent(1)-Y_Gradient_Descent).^2)/size(X_Gradient_Descent,2);
R2_GD=1-MES_GD/var(Y_Gradient_Descent)

%���淽�̷�
Y_Normal_Equation=data(:,35);
X_Normal_Equation=[ones(size(data(:,26),1),1),data(:,26)];
Theta_Normal_Equation=(X_Normal_Equation'*X_Normal_Equation)^-1*X_Normal_Equation'*Y_Normal_Equation;
MES_NE=sum((X_Normal_Equation(:,2)*Theta_Normal_Equation(2)+Theta_Normal_Equation(1)-Y_Normal_Equation).^2)/size(X_Normal_Equation,1);
R2_NE=1-MES_NE/var(Y_Normal_Equation)
%% ��������Իع�ģ��
%�ݶ��½���
Multi_variable=Feature_selection(1:5);
X_Gradient_Descent=data(:,Multi_variable)';
Y_Gradient_Descent=data(:,35)';
alpha=0.2;%��������Ӧ�ķ�����k/exp(k)+0.05
epoch=800;
detla=0.006;
[Theta_Gradient_Descent,Jcost]=liner_regression(X_Gradient_Descent,Y_Gradient_Descent,alpha,epoch,detla );
X_Gradient_Descent_R2=[ones(size(data(:,Multi_variable),1),1),data(:,Multi_variable)]';
MES_GD=sum((Theta_Gradient_Descent*X_Gradient_Descent_R2-Y_Gradient_Descent).^2)/size(X_Gradient_Descent,2);
R2_GD=1-MES_GD/var(Y_Gradient_Descent)

%���淽�̷�
Y_Normal_Equation=data(:,35);
X_Normal_Equation=[ones(size(data(:,Multi_variable),1),1),data(:,Multi_variable)];
Theta_Normal_Equation=(X_Normal_Equation'*X_Normal_Equation)^-1*X_Normal_Equation'*Y_Normal_Equation;
MES_NE=sum((Theta_Normal_Equation'*X_Normal_Equation'-Y_Normal_Equation').^2)/size(X_Normal_Equation,1);
R2_NE=1-MES_NE/var(Y_Normal_Equation)




