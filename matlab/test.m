A = rand(1000,1000);
B = rand(1000,500);
tic;
for i=1:100
a = A * B;
end
toc;