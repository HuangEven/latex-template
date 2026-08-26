function plot_qps_vs_gpu_scaling()
% 绘制图5-15：QPS 随 GPU 数量扩展曲线

root_dir = fileparts(mfilename('fullpath'));
font_name = 'Songti SC';
csv_path = fullfile(root_dir, 'gpu_scaling_qps.csv');
out_path = fullfile(root_dir, 'fig5_15_qps_vs_gpu.png');

tbl = readtable(csv_path, 'TextType', 'string');
if any(strcmp('scheme_desc', tbl.Properties.VariableNames))
    schemes = unique(tbl.scheme_desc, 'stable');
    scheme_field = "scheme_desc";
else
    schemes = unique(tbl.scheme, 'stable');
    scheme_field = "scheme";
end

fig = figure('Color', 'w', 'Position', [120, 120, 860, 520]);
ax = axes(fig);
hold(ax, 'on');

line_styles = {'-', '--', '-.', ':'};
markers = {'o', 's', '^', 'd'};

for i = 1:numel(schemes)
    mask = tbl.(scheme_field) == schemes(i);
    x = tbl.gpu_count(mask);
    y = tbl.qps(mask);

    plot(ax, x, y, ...
        'Color', 'k', ...
        'LineStyle', line_styles{min(i, numel(line_styles))}, ...
        'LineWidth', 1.6, ...
        'Marker', markers{min(i, numel(markers))}, ...
        'MarkerSize', 7, ...
        'MarkerFaceColor', 'w', ...
        'DisplayName', translate_scheme_label(schemes(i)));

    for j = 1:numel(x)
        text(x(j), y(j) + 180, sprintf('%.2f', y(j)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontName', font_name, ...
            'FontSize', 10);
    end
end

ax.FontName = font_name;
ax.FontSize = 12;
ax.LineWidth = 1.0;
ax.Box = 'off';
ax.TickDir = 'out';
ax.Layer = 'top';
ax.XGrid = 'on';
ax.YGrid = 'on';
ax.GridColor = [0.72, 0.72, 0.72];
ax.GridAlpha = 0.35;

xlim(ax, [0.8, 4.2]);
xticks(ax, [1, 2, 4]);
ylim(ax, [0, 15000]);

xlabel(ax, 'GPU数量', 'FontName', font_name, 'FontSize', 13);
ylabel(ax, '吞吐率（QPS）', 'FontName', font_name, 'FontSize', 13);
legend(ax, 'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off', 'FontName', font_name);

exportgraphics(fig, out_path, 'Resolution', 220);
close(fig);
end

function label = translate_scheme_label(value)
switch string(value)
    case {"Replicated primary path", "Replicated path"}
        label = "索引复制 / 数据并行";
    otherwise
        label = char(value);
end
end
