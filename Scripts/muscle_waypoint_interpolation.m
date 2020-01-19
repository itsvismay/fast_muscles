fname = '../data/contact_test/run5/input.json';
val = jsondecode(fileread(fname));
waypoints = val.muscle_waypoints;
n = fieldnames(waypoints);
h = 0.05;
waymatrix = cell2mat(struct2cell(waypoints));
fullmatrix = zeros(length(n), 1+(size(waymatrix,2)-1)/h);
names = {};
for i=1:length(n)
    field = n(i);
    names = [names field{1}];
    y = waymatrix(i,:);
    x = linspace(1, length(y), length(y));
    xx = x(1):h:x(length(y));
    yy = pchip(x, y, xx);
    fullmatrix(i,:) = yy;
end

T = array2table(fullmatrix', 'VariableNames', names);
S = table2struct(T);
val.muscle_starting_strength = S;

%disp(jsonencode(S))
jsontext = jsonencode(val);
jsontext = strrep(jsontext, ',', sprintf(',\n'));
jsontext = strrep(jsontext, '[{', sprintf('[\r{\r'));
jsontext = strrep(jsontext, '}]', sprintf('\r}\r]'));
fileID = fopen(fname, 'w');
fprintf(fileID, jsontext);
size(fullmatrix)
