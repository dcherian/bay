openfig('./images/wda-526-day-190-t0-120.fig')

fig = gcf();
H = findobj('-property', 'FontName');
for hh=H
    set(hh, 'FontName', 'Times')
    set(hh, 'FontSize', 13)
end



export_fig images/paper1/wda.pdf
% hax = findobj('-property', 'XLim');
% for ax = hax
%    set(hax, 'FontSize', 11)
% end
