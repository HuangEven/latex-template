function plot_latency_merge_compare()
% 绘制图5-16：延迟与归并方式对比图

root_dir = fileparts(mfilename('fullpath'));
font_name = 'Songti SC';
csv_path = fullfile(root_dir, 'latency_merge_compare.csv');
out_path = fullfile(root_dir, 'fig5_16_latency_merge_compare.png');

tbl = readtable(csv_path, 'TextType', 'string');
plot_values = [tbl.avg_latency_ms, tbl.p99_latency_ms];

fig = figure('Color', 'w', 'Position', [120, 120, 900, 520]);
ax = axes(fig);
hb = bar(ax, plot_values, 'grouped', 'BarWidth', 0.72);
hold(ax, 'on');

hb(1).FaceColor = [1, 1, 1];
hb(1).EdgeColor = [0, 0, 0];
hb(1).LineWidth = 1.2;

hb(2).FaceColor = [0.82, 0.82, 0.82];
hb(2).EdgeColor = [0, 0, 0];
hb(2).LineWidth = 1.2;

ax.FontName = font_name;
ax.FontSize = 12;
ax.LineWidth = 1.0;
ax.Box = 'off';
ax.TickDir = 'out';
ax.Layer = 'top';
ax.YGrid = 'on';
ax.GridColor = [0.72, 0.72, 0.72];
ax.GridAlpha = 0.35;

xticks(ax, 1:height(tbl));
if any(strcmp('scheme_desc', tbl.Properties.VariableNames))
    display_labels = wrap_labels(tbl.scheme_desc);
else
    display_labels = wrap_labels(tbl.scheme);
end
xticklabels(ax, display_labels);
ylim(ax, [0, 44]);

xlabel(ax, '归并方案', 'FontName', font_name, 'FontSize', 13);
ylabel(ax, '延迟（毫秒）', 'FontName', font_name, 'FontSize', 13);
legend(ax, {'平均延迟', 'p99延迟'}, 'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off', 'FontName', font_name);

for i = 1:numel(hb)
    x_points = hb(i).XEndPoints;
    y_points = hb(i).YEndPoints;
    for j = 1:numel(x_points)
        text(x_points(j), y_points(j) + 0.65, sprintf('%.2f', plot_values(j, i)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontName', font_name, ...
            'FontSize', 10);
    end
end

exportgraphics(fig, out_path, 'Resolution', 220);
close(fig);
end

function labels = wrap_labels(values)
labels = cell(size(values));
for i = 1:numel(values)
    label = string(values(i));
    switch label
        case {"Replicated primary path", "Replicated path"}
            labels{i} = "索引复制 /" + newline + "数据并行";
        case {"CPU-side aggregation fallback", "CPU aggregation"}
            labels{i} = "局部评分 +" + newline + "CPU侧聚合";
        case {"GPU-side merge fallback", "GPU merge"}
            labels{i} = "局部评分 +" + newline + "GPU侧归并";
        otherwise
            labels{i} = char(label);
    end
end
end
