function mat=convertVector2Mat(vector,m,n)
temp=reshape(vector,n,m);
mat=temp';