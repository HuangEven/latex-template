function plot_single_gpu_overview()
% 绘制图4-16：单卡性能综合对比图

root_dir = fileparts(mfilename('fullpath'));
font_name = 'Songti SC';
csv_path = fullfile(root_dir, 'single_gpu_overview_metrics.csv');
out_path = fullfile(root_dir, 'fig4_16_single_gpu_overview.png');

tbl = readtable(csv_path, 'TextType', 'string');
metric_fields = {'qps', 'p99_ms', 'pci_transfer_count', 'gpu_util'};
metric_labels = {'吞吐率（QPS）', '延迟（毫秒）', 'PCIe传输次数', 'GPU利用率（%）'};
metric_offsets = [90, 1.4, 0.14, 1.2];
metric_limits = [0 5200; 0 75; 0 7.0; 0 75];
scheme_colors = [
    1.00, 1.00, 1.00;
    0.82, 0.82, 0.82;
    0.64, 0.64, 0.64;
    0.42, 0.42, 0.42
];

fig = figure('Color', 'w', 'Position', [120, 120, 1120, 680]);
tiledlayout(fig, 2, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

if any(strcmp('scheme_desc', tbl.Properties.VariableNames))
    display_labels = wrap_labels(tbl.scheme_desc);
else
    display_labels = wrap_labels(tbl.scheme);
end

for i = 1:numel(metric_fields)
    ax = nexttile;
    values = tbl.(metric_fields{i});
    bars = bar(ax, values, 'BarWidth', 0.68, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.9);
    bars.CData = scheme_colors;

    ax.FontName = font_name;
    ax.FontSize = 12;
    ax.Box = 'off';
    ax.LineWidth = 1.0;
    ax.YGrid = 'on';
    ax.GridLineStyle = '--';
    ax.GridAlpha = 0.35;
    ax.Layer = 'top';
    ax.TickDir = 'out';

    xticks(ax, 1:height(tbl));
    xticklabels(ax, display_labels);
    xtickangle(ax, 0);
    xlabel(ax, '执行方案', 'FontName', font_name, 'FontSize', 13);
    ylabel(ax, metric_labels{i}, 'FontName', font_name, 'FontSize', 13);
    ylim(ax, metric_limits(i, :));

    for j = 1:numel(values)
        text(ax, j, values(j) + metric_offsets(i), sprintf('%.2f', values(j)), ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'bottom', ...
            'FontName', font_name, ...
            'FontSize', 10.0);
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
        case {"Original pipeline baseline", "Original baseline"}
            labels{i} = "原始链路" + newline + "基线";
        case {"Recall acceleration only", "Recall-accelerated"}
            labels{i} = "仅召回" + newline + "加速";
        case {"GPU-resident recall-to-score handoff", "GPU handoff"}
            labels{i} = "GPU侧常驻" + newline + "交接";
        case {"Full single-GPU optimization", "Full optimization"}
            labels{i} = "完整单卡" + newline + "优化";
        otherwise
            labels{i} = char(label);
    end
end
end
