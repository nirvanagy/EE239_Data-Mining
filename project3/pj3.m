%load data;
Data_movie = importdata('ml-100k/u.data');
user_id = Data_movie(:,1);
item_id = Data_movie(:,2);
rating = Data_movie(:,3);
addpath('nmfv1_4')

%_____________________________________________
r_matrix = zeros(943,1682);
w_matrix = zeros(943,1682);
for i = 1:100000
    r_matrix(user_id(i),item_id(i)) = rating(i);
    w_matrix(user_id(i),item_id(i)) = 1;
end 

k = [10,50,100];
%[A,Y,numIter,tElapsed,finalResidual]=wnmfrule(X,k,w,option)


%[U11,V11,numIter,tElapsed,finalResidual11]=newwnmfrule(r_matrix,k(1),w_matrix);
%[U12,V12,numIter,tElapsed,finalResidual12]=newwnmfrule(r_matrix,k(2),w_matrix);
%[U13,V13,numIter,tElapsed,finalResidual13]=newwnmfrule(r_matrix,k(3),w_matrix);

%__part2___??1?10fold??????????????????__??2:__????predicted?r_maxtrix??r_matrix2???________________________________________

random_index = randperm(100000);
r_matrix2 = zeros(943,1682);  % ???????for?????
w_matrix2 = zeros(943,1682);
r_predicted21 = zeros(943,1682,3);
error_matrix = zeros(10,3);
k =[10,50,100];

for j=1:10
    
    for i = 1:10       
        if i ~= j
            for a = ((i-1)*10000+1):(i*10000)
            r_matrix2(user_id(random_index(a)), item_id(random_index(a))) = rating(random_index(a));
            w_matrix2(user_id(random_index(a)), item_id(random_index(a))) = 1;                  
            end     
        end   
    end 
    for x = 1:3 
        [U21,V21,numIter,tElapsed,finalResidual21] = newwnmfrule(r_matrix2,k(x),w_matrix2);
        r_product = U21*V21;
        sum_of_error = 0;
    
        for i = ((j-1)*10000+1):j*10000
        
            user_id_j = user_id(random_index(i));
            item_id_j = item_id(random_index(i)); 
    
            sum_of_error = sum_of_error + abs(r_product(user_id_j, item_id_j) - r_matrix2(user_id_j, item_id_j));
            r_predicted21(user_id_j,item_id_j,x) = r_product(user_id_j, item_id_j);
        
        end 
    
        error_matrix(j,x) = sum_of_error/10000;
    end 
end 
error_matrix
mean(error_matrix(:,x))

%__3?______roc????_________________________________________________________________________________________

threshold = linspace(0.1, 5.0, 25);
precision_matrix = zeros(25,3);
recall_matrix    = zeros(25,3);
for i = 1:25
    for x = 1:3
        precision_matrix(i,x) = length(find(((r_predicted21(:,:,x))>(threshold(1,i)))&(r_matrix>3)))/length(find((r_predicted21(:,:,x))>(threshold(1,i))));
        recall_matrix(i,x)    = length(find(((r_predicted21(:,:,x))>(threshold(1,i)))&(r_matrix>3)))/length(find(r_matrix>3));
        
    end 
end

plot(threshold(:),(precision_matrix(:,1)/recall_matrix(:,1)));


%__4?_______________________________________________________________________________________________

r_matrix4 = w_matrix;
w_matrix4 = r_matrix;

k = [10,50,100];

[U41,V41,numIter,tElapsed,finalResidual41]=newwnmfrule(r_matrix4,k(1),w_matrix4);
[U42,V42,numIter,tElapsed,finalResidual42]=newwnmfrule(r_matrix4,k(2),w_matrix4);
[U43,V43,numIter,tElapsed,finalResidual43]=newwnmfrule(r_matrix4,k(3),w_matrix4);

least_squared_error4(1,1) = finalResidual41; 
least_squared_error4(2,1) = finalResidual42; 
least_squared_error4(3,1) = finalResidual43; 

r_matrix4ASL = w_matrix4;
w_matrix4ASL = r_matrix4;
lambda = [0.01,0.1,1];
k = [10,50,100];

lse4ASL=zeros(3,3);
for i = 1:3;
    [U41ASL,V41ASL,numIter,tElapsed,finalResidual41ASL]=new2wnmfrule(r_matrix4ASL,k(1),w_matrix4ASL,lambda(i));
    [U42ASL,V42ASL,numIter,tElapsed,finalResidual42ASL]=new2wnmfrule(r_matrix4ASL,k(2),w_matrix4ASL,lambda(i));
    [U43ASL,V43ASL,numIter,tElapsed,finalResidual43ASL]=new2wnmfrule(r_matrix4ASL,k(3),w_matrix4ASL,lambda(i));

    lse4ASL(1,i) = finalResidual41ASL; 
    lse4ASL(2,i) = finalResidual42ASL; 
    lse4ASL(3,i) = finalResidual43ASL; 
end 


