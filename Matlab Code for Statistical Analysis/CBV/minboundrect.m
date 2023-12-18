function [rectx,recty,area,perimeter] = minboundrect(x,y,metric)

if (nargin<3) || isempty(metric)
  metric = 'a';
elseif ~ischar(metric)
  error 'metric must be a character flag if it is supplied.'
else
 
  metric = lower(metric(:)');                    
  ind = strmatch(metric,{'area','perimeter'});             
  if isempty(ind)                
    error 'metric does not match either ''area'' or ''perimeter'''
  end

  
  metric = metric(1);
end


x=x(:);
y=y(:);

n = length(x);                                    
if n~=length(y)                               
  error 'x and y must be the same sizes'
end


if n>3 


    if (var(x)== 0|| var(y)==0)
        if var(x)== 0
            x = [x-1;x(1); x+1 ];
            y = [y ;y(1);y];
            flag = 1;
        else
            y = [y-1;y(1); y+1 ];
            x = [x ;x(1);x];
            flag = 1;
        end

    else
        flag = 0;
    edges = convhull(x,y); 

    end

  if flag == 0 
  
  x = x(edges);
  y = y(edges);
  
  end
 
  nedges = length(x) - 1;                       
elseif n>1
  
  nedges = n;
  x(end+1) = x(1);
  y(end+1) = y(1);
else
  
  nedges = n;
end

switch nedges
  case 0
    
    rectx = [];
    recty = [];
    area = [];
    perimeter = [];
    return
  case 1
    
    rectx = repmat(x,1,5);
    recty = repmat(y,1,5);
    area = 0;
    perimeter = 0;
    return
  case 2
    
    rectx = x([1 2 2 1 1]);
    recty = y([1 2 2 1 1]);
    area = 0;
    perimeter = 2*sqrt(diff(x).^2 + diff(y).^2);
    return
end

Rmat = @(theta) [cos(theta) sin(theta);-sin(theta) cos(theta)];

ind = 1:(length(x)-1);
edgeangles = atan2(y(ind+1) - y(ind),x(ind+1) - x(ind));

edgeangles = unique(mod(edgeangles,pi/2));

nang = length(edgeangles);              
area = inf;                           
perimeter = inf;
met = inf;
xy = [x,y];
for i = 1:nang                         
  rot = Rmat(-edgeangles(i));
  xyr = xy*rot;
  xymin = min(xyr,[],1);
  xymax = max(xyr,[],1);

  A_i = prod(xymax - xymin);
  P_i = 2*sum(xymax-xymin);

  if metric=='a'
    M_i = A_i;
  else
    M_i = P_i;
  end

  if M_i<met
    
    met = M_i;
    area = A_i;
    perimeter = P_i;

    rect = [xymin;[xymax(1),xymin(2)];xymax;[xymin(1),xymax(2)];xymin];
    rect = rect*rot';
    rectx = rect(:,1);
    recty = rect(:,2);
  end
end

end 