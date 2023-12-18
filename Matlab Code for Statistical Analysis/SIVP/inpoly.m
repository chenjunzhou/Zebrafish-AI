function [cn,on] = inpoly(p,node,edge,TOL)
if nargin<4
   TOL = 1.0e-12;
   if nargin<3
      edge = [];
      if nargin<2
         error('Insufficient inputs');
      end
   end
end
nnode = size(node,1);
if isempty(edge)                                                           
   edge = [(1:nnode-1)' (2:nnode)'; nnode 1];
end
if size(p,2)~=2
   error('P must be an Nx2 array.');
end
if size(node,2)~=2
   error('NODE must be an Mx2 array.');
end
if size(edge,2)~=2
   error('EDGE must be an Mx2 array.');
end
if max(edge(:))>nnode || any(edge(:)<1)
   error('Invalid EDGE.');
end
n  = size(p,1);
nc = size(edge,1);

dxy = max(p,[],1)-min(p,[],1);
if dxy(1)>dxy(2)
  
   p = p(:,[2,1]);
   node = node(:,[2,1]);
end
tol = TOL*min(dxy);

% Sort test points by y-value
[y,i] = sort(p(:,2));
x = p(i,1);

cn = false(n,1);    
                     
                     
on = cn;
for k = 1:nc         

   n1 = edge(k,1);
   n2 = edge(k,2);

   y1 = node(n1,2);
   y2 = node(n2,2);
   if y1<y2
      x1 = node(n1,1);
      x2 = node(n2,1);
   else
      yt = y1;
      y1 = y2;
      y2 = yt;
      x1 = node(n2,1);
      x2 = node(n1,1);
   end
   if x1>x2
      xmin = x2;
      xmax = x1;
   else
      xmin = x1;
      xmax = x2;
   end

   if y(1)>=y1
      start = 1;
   elseif y(n)<y1
      start = n+1;       
   else
      lower = 1;
      upper = n;
      for j = 1:n
         start = round(0.5*(lower+upper));
         if y(start)<y1
            lower = start;
         elseif y(start-1)<y1
            break;
         else
            upper = start;
         end
      end
   end

   for j = start:n
    

      Y = y(j);  
      if Y<=y2
         X = x(j);  
         if X>=xmin
            if X<=xmax

               on(j) = on(j) || (abs((y2-Y)*(x1-X)-(y1-Y)*(x2-X))<tol);

               if (Y<y2) && ((y2-y1)*(X-x1)<(Y-y1)*(x2-x1))
                  cn(j) = ~cn(j);
               end

            end
         elseif Y<y2   
            cn(j) = ~cn(j);
         end
      else
         break
      end
   end

end
cn(i) = cn|on;
on(i) = on;

end     