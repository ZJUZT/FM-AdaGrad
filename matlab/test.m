tic;
[idx,C] = kmeans(sparse(train_X),100,'Display','iter','Replicate',5);
toc;