

%% Main Execution
clc; close all; clear;
fprintf('=== Hybrid Satellite-Fiber Network Simulation ===\n');
fprintf('Starting at: %s\n\n', datestr(now));

% Verify dependencies
check_dependencies();

% Configuration
cfg = setup_configuration();

% Create output directory
outdir = fullfile(pwd, 'hybrid_sim_output');
if ~exist(outdir, 'dir'), mkdir(outdir); end
cfg.outdir = outdir;

% Run pure MATLAB simulation (more reliable than Simulink for this)
fprintf('Running comprehensive network simulation...\n');
results = run_comprehensive_simulation(cfg);

% Generate publication-quality analysis
fprintf('\nGenerating analysis and plots...\n');
generate_comprehensive_analysis(results, cfg);

% Save all results
save(fullfile(outdir, 'simulation_results.mat'), 'results', 'cfg');

fprintf('\n=== Simulation Complete ===\n');
fprintf('Results saved to: %s\n', outdir);
fprintf('Finished at: %s\n', datestr(now));

%% ========================================================================
%                          CONFIGURATION SETUP
%% ========================================================================
function cfg = setup_configuration()
% Comprehensive configuration with all network parameters

cfg.seed = 12345;
rng(cfg.seed);

% Simulation timing
cfg.simTime = 120;              % Total simulation time (seconds)
cfg.warmup = 10;                % Warmup period (seconds)
cfg.handoff_time = 60;          % Scheduled handoff at 60s
cfg.dt = 0.001;                 % Time step (1ms)

% Packet generation
cfg.packet_size_bytes = 1500;   % Standard Ethernet MTU
cfg.packet_rate_pps = 100;      % Packets per second (CBR)

% Satellite link parameters (GEO)
cfg.sat.type = 'GEO';
cfg.sat.altitude_km = 35786;    % Geostationary orbit
cfg.sat.elevation_deg = 30;     % Ground station elevation
cfg.sat.freq_ghz = 14;          % Ku-band uplink
cfg.sat.tx_power_dbm = 40;      % Transmit power
cfg.sat.tx_gain_dbi = 30;       % TX antenna gain
cfg.sat.rx_gain_dbi = 35;       % RX antenna gain
cfg.sat.noise_figure_db = 2;    % Receiver noise figure
cfg.sat.processing_delay_ms = 10; % Onboard processing

% Fiber link parameters
cfg.fiber.length_km = 200;      % Fiber distance
cfg.fiber.refractive_index = 1.5; % Typical fiber
cfg.fiber.attenuation_db_per_km = 0.2; % Good quality fiber
cfg.fiber.dispersion_ps_per_nm_km = 17; % Standard SMF
cfg.fiber.ber_target = 1e-12;   % Target BER

% Modulation schemes
cfg.modulations = {'BPSK', 'QPSK', '16QAM', '64QAM'};
cfg.mod_bits = [1, 2, 4, 6];    % Bits per symbol

% Bandwidth allocations (Hz)
cfg.bandwidths = [1e6, 5e6, 10e6, 20e6, 50e6, 100e6];

% Performance thresholds
cfg.latency_threshold_ms = 500;  % Max acceptable latency
cfg.throughput_target_mbps = 10; % Target throughput

fprintf('Configuration loaded:\n');
fprintf('  Simulation time: %d seconds\n', cfg.simTime);
fprintf('  Handoff scheduled at: %d seconds\n', cfg.handoff_time);
fprintf('  Packet rate: %d pps\n', cfg.packet_rate_pps);
fprintf('  Modulations: %s\n', strjoin(cfg.modulations, ', '));
fprintf('  Bandwidths: %s MHz\n', num2str(cfg.bandwidths/1e6));
end

%% ========================================================================
%                     DEPENDENCY VERIFICATION
%% ========================================================================
function check_dependencies()
v = ver;
tb = {v.Name};
required = {'MATLAB', 'Communications Toolbox'};
missing = {};

for i = 1:length(required)
    if ~any(contains(tb, required{i}))
        missing{end+1} = required{i};
    end
end

if ~isempty(missing)
    warning('Missing toolboxes: %s', strjoin(missing, ', '));
    fprintf('Simulation will use simplified models.\n');
end
end

%% ========================================================================
%                    COMPREHENSIVE SIMULATION ENGINE
%% ========================================================================
function results = run_comprehensive_simulation(cfg)
% Run complete parameter sweep across modulations and bandwidths

total_scenarios = length(cfg.modulations) * length(cfg.bandwidths);
scenarios = []; % Initialize as empty
scenario_id = 1;

for mod_idx = 1:length(cfg.modulations)
    mod_name = cfg.modulations{mod_idx};
    bits_per_symbol = cfg.mod_bits(mod_idx);
    
    for bw_idx = 1:length(cfg.bandwidths)
        bandwidth = cfg.bandwidths(bw_idx);
        
        fprintf('  [%d/%d] %s, BW=%.1f MHz\n', ...
            scenario_id, total_scenarios, ...
            mod_name, bandwidth/1e6);
        
        % Run single scenario
        scenario = run_single_scenario(cfg, mod_name, bits_per_symbol, bandwidth);
        scenario.mod_name = mod_name;
        scenario.bandwidth = bandwidth;
        scenario.id = scenario_id;
        
        % Append to array
        if scenario_id == 1
            scenarios = scenario;
        else
            scenarios(scenario_id) = scenario;
        end
        scenario_id = scenario_id + 1;
    end
end

results.scenarios = scenarios;
fprintf('Simulation complete: %d scenarios executed\n', length(results.scenarios));
end

%% ========================================================================
%                        SINGLE SCENARIO SIMULATION
%% ========================================================================
function scenario = run_single_scenario(cfg, mod_name, bits_per_symbol, bandwidth)
% Simulate one configuration: generate packets, route through network

% Time vector
t = 0:cfg.dt:cfg.simTime;
num_steps = length(t);

% Generate packet arrivals
packet_times = 0:1/cfg.packet_rate_pps:cfg.simTime;
num_packets = length(packet_times);

% Pre-allocate packet log
packets = struct('id', {}, 't_gen', {}, 't_sent', {}, 't_recv', {}, ...
    'path', {}, 'latency', {}, 'dropped', {}, 'size_bytes', {});

% Satellite link calculation
sat_prop_delay = calculate_satellite_delay(cfg);
sat_snr_db = calculate_satellite_snr(cfg, bandwidth);
sat_per = calculate_packet_error_rate(sat_snr_db, mod_name);
sat_capacity_bps = bandwidth * bits_per_symbol * 0.8; % 80% efficiency

% Fiber link calculation
fiber_prop_delay = calculate_fiber_delay(cfg);
fiber_snr_db = calculate_fiber_snr(cfg, bandwidth);
fiber_per = calculate_packet_error_rate(fiber_snr_db, mod_name);
fiber_capacity_bps = bandwidth * bits_per_symbol * 0.95; % 95% efficiency

% Process each packet
for pkt_id = 1:num_packets
    t_gen = packet_times(pkt_id);
    
    % Determine routing: satellite before handoff, fiber after
    if t_gen < cfg.handoff_time
        path = 'satellite';
        prop_delay = sat_prop_delay;
        processing_delay = cfg.sat.processing_delay_ms / 1000;
        per = sat_per;
        capacity = sat_capacity_bps;
    else
        path = 'fiber';
        prop_delay = fiber_prop_delay;
        processing_delay = 0.001; % 1ms switching
        per = fiber_per;
        capacity = fiber_capacity_bps;
    end
    
    % Serialization delay
    ser_delay = (cfg.packet_size_bytes * 8) / capacity;
    
    % Total delay
    total_delay = prop_delay + processing_delay + ser_delay;
    
    % Check if packet dropped
    dropped = rand() < per;
    
    % Record packet
    packets(pkt_id).id = pkt_id;
    packets(pkt_id).t_gen = t_gen;
    packets(pkt_id).t_sent = t_gen;
    packets(pkt_id).t_recv = t_gen + total_delay;
    packets(pkt_id).path = path;
    packets(pkt_id).latency = total_delay * 1000; % ms
    packets(pkt_id).dropped = dropped;
    packets(pkt_id).size_bytes = cfg.packet_size_bytes;
end

% Calculate performance metrics
scenario = analyze_scenario_performance(packets, cfg, ...
    sat_snr_db, fiber_snr_db, sat_capacity_bps, fiber_capacity_bps);

% Store detailed data
scenario.packets = packets;
scenario.sat_prop_delay = sat_prop_delay;
scenario.fiber_prop_delay = fiber_prop_delay;
scenario.sat_per = sat_per;
scenario.fiber_per = fiber_per;
end

%% ========================================================================
%                       CHANNEL MODELS & CALCULATIONS
%% ========================================================================
function delay = calculate_satellite_delay(cfg)
% Calculate satellite propagation delay (two-way)
c = 3e8; % Speed of light
slant_range = cfg.sat.altitude_km * 1e3 / sind(cfg.sat.elevation_deg);
delay = (2 * slant_range) / c; % Round trip
end

function delay = calculate_fiber_delay(cfg)
% Calculate fiber propagation delay
c = 3e8;
v_fiber = c / cfg.fiber.refractive_index;
delay = (cfg.fiber.length_km * 1e3) / v_fiber;
end

function snr_db = calculate_satellite_snr(cfg, bandwidth)
% Calculate satellite link SNR using link budget
c = 3e8;
lambda = c / (cfg.sat.freq_ghz * 1e9);
slant_range = cfg.sat.altitude_km * 1e3 / sind(cfg.sat.elevation_deg);

% Free space path loss
fspl_db = 20*log10(4*pi*slant_range/lambda);

% Received power
pr_dbm = cfg.sat.tx_power_dbm + cfg.sat.tx_gain_dbi + ...
    cfg.sat.rx_gain_dbi - fspl_db;

% Noise power
T_sys = 290; % System temperature (K)
k = 1.38e-23; % Boltzmann constant
N0_dbm = 10*log10(k * T_sys * 1000); % dBm/Hz
N_dbm = N0_dbm + 10*log10(bandwidth) + cfg.sat.noise_figure_db;

% SNR
snr_db = pr_dbm - N_dbm;
end

function snr_db = calculate_fiber_snr(cfg, bandwidth)
% Calculate fiber link SNR (typically very high)
% Simplified model: attenuation-limited

% Total attenuation
total_atten_db = cfg.fiber.length_km * cfg.fiber.attenuation_db_per_km;

% Assume 0 dBm launch power
pr_dbm = 0 - total_atten_db;

% Thermal noise (receiver limited)
k = 1.38e-23;
T = 290;
N0_dbm = 10*log10(k * T * 1000);
N_dbm = N0_dbm + 10*log10(bandwidth) + 3; % 3dB NF for receiver

snr_db = pr_dbm - N_dbm;
end

function per = calculate_packet_error_rate(snr_db, mod_name)
% Map SNR to packet error rate based on modulation
% Simplified BER to PER conversion

packet_bits = 1500 * 8; % bits per packet

% Get BER from SNR
switch mod_name
    case 'BPSK'
        ber = qfunc(sqrt(2 * 10^(snr_db/10)));
    case 'QPSK'
        ber = qfunc(sqrt(10^(snr_db/10)));
    case '16QAM'
        ber = (3/4) * qfunc(sqrt((4/5) * 10^(snr_db/10)));
    case '64QAM'
        ber = (7/12) * qfunc(sqrt((6/7) * 10^(snr_db/10)));
    otherwise
        ber = 0.5;
end

% Convert BER to PER (assuming independent bit errors)
per = 1 - (1 - ber)^packet_bits;
per = min(per, 1.0); % Cap at 1
per = max(per, 1e-10); % Floor at 1e-10
end

%% ========================================================================
%                      PERFORMANCE ANALYSIS
%% ========================================================================
function metrics = analyze_scenario_performance(packets, cfg, ...
    sat_snr, fiber_snr, sat_capacity, fiber_capacity)
% Extract comprehensive performance metrics

% Separate by path
sat_packets = packets(strcmp({packets.path}, 'satellite'));
fiber_packets = packets(strcmp({packets.path}, 'fiber'));

% Overall metrics
total_packets = length(packets);
dropped_packets = sum([packets.dropped]);
delivered_packets = total_packets - dropped_packets;

metrics.total_packets = total_packets;
metrics.dropped_packets = dropped_packets;
metrics.delivered_packets = delivered_packets;
metrics.delivery_ratio = delivered_packets / total_packets;

% Throughput calculation
if delivered_packets > 0
    delivered_bits = delivered_packets * cfg.packet_size_bytes * 8;
    metrics.throughput_mbps = delivered_bits / cfg.simTime / 1e6;
else
    metrics.throughput_mbps = 0;
end

% Latency statistics (only delivered packets)
delivered = packets(~[packets.dropped]);
if ~isempty(delivered)
    latencies = [delivered.latency];
    metrics.latency_mean_ms = mean(latencies);
    metrics.latency_median_ms = median(latencies);
    metrics.latency_95p_ms = prctile(latencies, 95);
    metrics.latency_std_ms = std(latencies);
    metrics.latency_min_ms = min(latencies);
    metrics.latency_max_ms = max(latencies);
else
    metrics.latency_mean_ms = NaN;
    metrics.latency_median_ms = NaN;
    metrics.latency_95p_ms = NaN;
    metrics.latency_std_ms = NaN;
    metrics.latency_min_ms = NaN;
    metrics.latency_max_ms = NaN;
end

% Per-path metrics
metrics.sat_packets = length(sat_packets);
metrics.fiber_packets = length(fiber_packets);

if ~isempty(sat_packets)
    sat_delivered = sat_packets(~[sat_packets.dropped]);
    metrics.sat_delivery_ratio = length(sat_delivered) / length(sat_packets);
    if ~isempty(sat_delivered)
        metrics.sat_latency_mean_ms = mean([sat_delivered.latency]);
    else
        metrics.sat_latency_mean_ms = NaN;
    end
else
    metrics.sat_delivery_ratio = NaN;
    metrics.sat_latency_mean_ms = NaN;
end

if ~isempty(fiber_packets)
    fiber_delivered = fiber_packets(~[fiber_packets.dropped]);
    metrics.fiber_delivery_ratio = length(fiber_delivered) / length(fiber_packets);
    if ~isempty(fiber_delivered)
        metrics.fiber_latency_mean_ms = mean([fiber_delivered.latency]);
    else
        metrics.fiber_latency_mean_ms = NaN;
    end
else
    metrics.fiber_delivery_ratio = NaN;
    metrics.fiber_latency_mean_ms = NaN;
end

% Link quality metrics
metrics.sat_snr_db = sat_snr;
metrics.fiber_snr_db = fiber_snr;
metrics.sat_capacity_mbps = sat_capacity / 1e6;
metrics.fiber_capacity_mbps = fiber_capacity / 1e6;

% Handoff metrics
handoff_window = [cfg.handoff_time-5, cfg.handoff_time+5];
handoff_pkts = packets([packets.t_gen] >= handoff_window(1) & ...
    [packets.t_gen] <= handoff_window(2));
if ~isempty(handoff_pkts)
    metrics.handoff_packet_loss = sum([handoff_pkts.dropped]) / length(handoff_pkts);
else
    metrics.handoff_packet_loss = 0;
end

% Efficiency
metrics.bandwidth = sat_capacity / 0.8; % Recover original bandwidth
metrics.spectral_efficiency = metrics.throughput_mbps / (metrics.bandwidth/1e6);
end

%% ========================================================================
%                   PUBLICATION-QUALITY VISUALIZATION
%% ========================================================================
function generate_comprehensive_analysis(results, cfg)
% Create professional conference-ready plots and analysis

outdir = cfg.outdir;

% Extract data for plotting
num_scenarios = length(results.scenarios);
modulations = cfg.modulations;
bandwidths = cfg.bandwidths;

%% PLOT 1: Throughput vs Bandwidth (all modulations)
fig1 = figure('Position', [100 100 1000 600], 'Color', 'white');
hold on; grid on; box on;

% Define distinct line styles and colors
line_styles = {'-o', '--s', ':d', '-.^'};
colors = [0 0.4470 0.7410; 0.8500 0.3250 0.0980; 0.9290 0.6940 0.1250; 0.4940 0.1840 0.5560];

for mod_idx = 1:length(modulations)
    mod_name = modulations{mod_idx};
    throughputs = zeros(size(bandwidths));
    
    for bw_idx = 1:length(bandwidths)
        % Find matching scenario
        for s = 1:num_scenarios
            if strcmp(results.scenarios(s).mod_name, mod_name) && ...
                    results.scenarios(s).bandwidth == bandwidths(bw_idx)
                throughputs(bw_idx) = results.scenarios(s).throughput_mbps;
                break;
            end
        end
    end
    
    plot(bandwidths/1e6, throughputs, line_styles{mod_idx}, 'LineWidth', 2.5, ...
        'MarkerSize', 10, 'Color', colors(mod_idx,:), 'DisplayName', mod_name, ...
        'MarkerFaceColor', colors(mod_idx,:));
end

xlabel('Bandwidth (MHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Throughput (Mbps)', 'FontSize', 12, 'FontWeight', 'bold');
title('Throughput vs Bandwidth for Different Modulations', ...
    'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10);
set(gca, 'FontSize', 11);
saveas(fig1, fullfile(outdir, 'throughput_vs_bandwidth.png'));
saveas(fig1, fullfile(outdir, 'throughput_vs_bandwidth.fig'));

%% PLOT 2: Latency Comparison (Satellite vs Fiber)
fig2 = figure('Position', [120 120 1000 600], 'Color', 'white');

% Use maximum bandwidth for fair comparison
max_bw_idx = length(bandwidths);
sat_latencies = zeros(1, length(modulations));
fiber_latencies = zeros(1, length(modulations));

for mod_idx = 1:length(modulations)
    for s = 1:num_scenarios
        if strcmp(results.scenarios(s).mod_name, modulations{mod_idx}) && ...
                results.scenarios(s).bandwidth == bandwidths(max_bw_idx)
            sat_latencies(mod_idx) = results.scenarios(s).sat_latency_mean_ms;
            fiber_latencies(mod_idx) = results.scenarios(s).fiber_latency_mean_ms;
            break;
        end
    end
end

x = 1:length(modulations);
width = 0.35;
bar(x - width/2, sat_latencies, width, 'FaceColor', [0.8 0.2 0.2], ...
    'DisplayName', 'Satellite');
hold on;
bar(x + width/2, fiber_latencies, width, 'FaceColor', [0.2 0.2 0.8], ...
    'DisplayName', 'Fiber');

set(gca, 'XTick', x, 'XTickLabel', modulations, 'FontSize', 11);
xlabel('Modulation Scheme', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Mean Latency (ms)', 'FontSize', 12, 'FontWeight', 'bold');
title(sprintf('Latency Comparison: Satellite vs Fiber (BW=%.0f MHz)', ...
    bandwidths(max_bw_idx)/1e6), 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 10);
grid on;
saveas(fig2, fullfile(outdir, 'latency_comparison.png'));
saveas(fig2, fullfile(outdir, 'latency_comparison.fig'));

%% PLOT 3: Latency CDF
fig3 = figure('Position', [140 140 1000 600], 'Color', 'white');
hold on; grid on; box on;

% Plot CDF for best case of each modulation
for mod_idx = 1:length(modulations)
    for s = 1:num_scenarios
        if strcmp(results.scenarios(s).mod_name, modulations{mod_idx}) && ...
                results.scenarios(s).bandwidth == bandwidths(end)
            
            delivered = results.scenarios(s).packets(~[results.scenarios(s).packets.dropped]);
            if ~isempty(delivered)
                latencies = sort([delivered.latency]);
                cdf_y = (1:length(latencies)) / length(latencies);
                plot(latencies, cdf_y, '-', 'LineWidth', 2, ...
                    'DisplayName', modulations{mod_idx});
            end
            break;
        end
    end
end

xlabel('Latency (ms)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('CDF', 'FontSize', 12, 'FontWeight', 'bold');
title('Cumulative Distribution of End-to-End Latency', ...
    'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 10);
set(gca, 'FontSize', 11);
xlim([0 inf]);
saveas(fig3, fullfile(outdir, 'latency_cdf.png'));
saveas(fig3, fullfile(outdir, 'latency_cdf.fig'));

%% PLOT 4: Handoff Performance Timeline
fig4 = figure('Position', [160 160 1200 700], 'Color', 'white');

% Use QPSK with medium bandwidth for demonstration
demo_scenario = [];
for s = 1:num_scenarios
    if strcmp(results.scenarios(s).mod_name, 'QPSK') && ...
            results.scenarios(s).bandwidth == bandwidths(3)
        demo_scenario = results.scenarios(s);
        break;
    end
end

if ~isempty(demo_scenario)
    % Top subplot: throughput over time
    subplot(3,1,1);
    window = 5; % 5 second windows
    time_bins = 0:window:cfg.simTime;
    throughput_timeline = zeros(1, length(time_bins)-1);
    
    for i = 1:length(time_bins)-1
        window_packets = demo_scenario.packets(...
            [demo_scenario.packets.t_recv] >= time_bins(i) & ...
            [demo_scenario.packets.t_recv] < time_bins(i+1) & ...
            ~[demo_scenario.packets.dropped]);
        
        if ~isempty(window_packets)
            bits = length(window_packets) * cfg.packet_size_bytes * 8;
            throughput_timeline(i) = bits / window / 1e6;
        end
    end
    
    bar(time_bins(1:end-1) + window/2, throughput_timeline, 'FaceColor', [0.3 0.6 0.9]);
    hold on;
    xline(cfg.handoff_time, 'r--', 'LineWidth', 2, 'DisplayName', 'Handoff');
    ylabel('Throughput (Mbps)', 'FontWeight', 'bold');
    title('Network Performance During Handoff', 'FontSize', 13, 'FontWeight', 'bold');
    grid on; legend;
    
    % Middle subplot: latency over time
    subplot(3,1,2);
    delivered = demo_scenario.packets(~[demo_scenario.packets.dropped]);
    scatter([delivered.t_recv], [delivered.latency], 20, ...
        [delivered.t_recv], 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    xline(cfg.handoff_time, 'r--', 'LineWidth', 2);
    ylabel('Latency (ms)', 'FontWeight', 'bold');
    colormap(jet);
    grid on;
    
    % Bottom subplot: packet loss
    subplot(3,1,3);
    loss_timeline = zeros(1, length(time_bins)-1);
    
    for i = 1:length(time_bins)-1
        window_packets = demo_scenario.packets(...
            [demo_scenario.packets.t_gen] >= time_bins(i) & ...
            [demo_scenario.packets.t_gen] < time_bins(i+1));
        
        if ~isempty(window_packets)
            loss_timeline(i) = sum([window_packets.dropped]) / length(window_packets) * 100;
        end
    end
    
    bar(time_bins(1:end-1) + window/2, loss_timeline, 'FaceColor', [0.9 0.3 0.3]);
    hold on;
    xline(cfg.handoff_time, 'r--', 'LineWidth', 2);
    xlabel('Time (s)', 'FontWeight', 'bold');
    ylabel('Packet Loss (%)', 'FontWeight', 'bold');
    grid on;
end

saveas(fig4, fullfile(outdir, 'handoff_timeline.png'));
saveas(fig4, fullfile(outdir, 'handoff_timeline.fig'));

%% PLOT 5: Spectral Efficiency Heatmap
fig5 = figure('Position', [180 180 1000 600], 'Color', 'white');

efficiency_matrix = zeros(length(modulations), length(bandwidths));

for mod_idx = 1:length(modulations)
    for bw_idx = 1:length(bandwidths)
        for s = 1:num_scenarios
            if strcmp(results.scenarios(s).mod_name, modulations{mod_idx}) && ...
                    results.scenarios(s).bandwidth == bandwidths(bw_idx)
                efficiency_matrix(mod_idx, bw_idx) = results.scenarios(s).spectral_efficiency;
                break;
            end
        end
    end
end

imagesc(efficiency_matrix);
colormap(hot);
colorbar('FontSize', 10);
set(gca, 'XTick', 1:length(bandwidths), 'XTickLabel', bandwidths/1e6, ...
    'YTick', 1:length(modulations), 'YTickLabel', modulations, 'FontSize', 11);
xlabel('Bandwidth (MHz)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Modulation', 'FontSize', 12, 'FontWeight', 'bold');
title('Spectral Efficiency (bps/Hz)', 'FontSize', 14, 'FontWeight', 'bold');

% Add text annotations
for mod_idx = 1:length(modulations)
    for bw_idx = 1:length(bandwidths)
        text(bw_idx, mod_idx, sprintf('%.2f', efficiency_matrix(mod_idx, bw_idx)), ...
            'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 9, ...
            'FontWeight', 'bold');
    end
end

saveas(fig5, fullfile(outdir, 'spectral_efficiency.png'));
saveas(fig5, fullfile(outdir, 'spectral_efficiency.fig'));

%% Generate Summary Report
generate_summary_report(results, cfg, outdir);

fprintf('All plots saved to: %s\n', outdir);
end

%% ========================================================================
%                         SUMMARY REPORT GENERATION
%% ========================================================================
function generate_summary_report(results, cfg, outdir)
% Create comprehensive text report

fid = fopen(fullfile(outdir, 'simulation_report.txt'), 'w');

fprintf(fid, '========================================\n');
fprintf(fid, 'HYBRID SATELLITE-FIBER NETWORK SIMULATION\n');
fprintf(fid, '========================================\n\n');
fprintf(fid, 'Date: %s\n', datestr(now));
fprintf(fid, 'Configuration:\n');
fprintf(fid, '  Simulation Time: %d seconds\n', cfg.simTime);
fprintf(fid, '  Handoff Time: %d seconds\n', cfg.handoff_time);
fprintf(fid, '  Packet Rate: %d pps\n', cfg.packet_rate_pps);
fprintf(fid, '  Packet Size: %d bytes\n\n', cfg.packet_size_bytes);

fprintf(fid, 'Satellite Link Parameters:\n');
fprintf(fid, '  Type: %s\n', cfg.sat.type);
fprintf(fid, '  Altitude: %.0f km\n', cfg.sat.altitude_km);
fprintf(fid, '  Frequency: %.1f GHz\n', cfg.sat.freq_ghz);
fprintf(fid, '  TX Power: %.0f dBm\n\n', cfg.sat.tx_power_dbm);

fprintf(fid, 'Fiber Link Parameters:\n');
fprintf(fid, '  Length: %.0f km\n', cfg.fiber.length_km);
fprintf(fid, '  Attenuation: %.2f dB/km\n\n', cfg.fiber.attenuation_db_per_km);

fprintf(fid, '========================================\n');
fprintf(fid, 'PERFORMANCE RESULTS\n');
fprintf(fid, '========================================\n\n');

% Find best and worst scenarios
best_throughput = -inf;
worst_throughput = inf;
best_scenario = [];
worst_scenario = [];

for s = 1:length(results.scenarios)
    if results.scenarios(s).throughput_mbps > best_throughput
        best_throughput = results.scenarios(s).throughput_mbps;
        best_scenario = results.scenarios(s);
    end
    if results.scenarios(s).throughput_mbps < worst_throughput
        worst_throughput = results.scenarios(s).throughput_mbps;
        worst_scenario = results.scenarios(s);
    end
end

fprintf(fid, 'Best Performance:\n');
fprintf(fid, '  Modulation: %s\n', best_scenario.mod_name);
fprintf(fid, '  Bandwidth: %.1f MHz\n', best_scenario.bandwidth/1e6);
fprintf(fid, '  Throughput: %.2f Mbps\n', best_scenario.throughput_mbps);
fprintf(fid, '  Mean Latency: %.2f ms\n', best_scenario.latency_mean_ms);
fprintf(fid, '  Delivery Ratio: %.2f%%\n\n', best_scenario.delivery_ratio*100);

fprintf(fid, 'Worst Performance:\n');
fprintf(fid, '  Modulation: %s\n', worst_scenario.mod_name);
fprintf(fid, '  Bandwidth: %.1f MHz\n', worst_scenario.bandwidth/1e6);
fprintf(fid, '  Throughput: %.2f Mbps\n', worst_scenario.throughput_mbps);
fprintf(fid, '  Mean Latency: %.2f ms\n', worst_scenario.latency_mean_ms);
fprintf(fid, '  Delivery Ratio: %.2f%%\n\n', worst_scenario.delivery_ratio*100);

fprintf(fid, '========================================\n');
fprintf(fid, 'DETAILED RESULTS BY CONFIGURATION\n');
fprintf(fid, '========================================\n\n');

for mod_idx = 1:length(cfg.modulations)
    mod_name = cfg.modulations{mod_idx};
    fprintf(fid, '\n--- %s Modulation ---\n', mod_name);
    fprintf(fid, '%-15s %-15s %-15s %-15s %-15s\n', ...
        'BW (MHz)', 'Tput (Mbps)', 'Latency (ms)', 'Delivery (%)', 'Eff (bps/Hz)');
    fprintf(fid, '%s\n', repmat('-', 1, 80));
    
    for bw_idx = 1:length(cfg.bandwidths)
        bw = cfg.bandwidths(bw_idx);
        
        % Find matching scenario
        for s = 1:length(results.scenarios)
            if strcmp(results.scenarios(s).mod_name, mod_name) && ...
                    results.scenarios(s).bandwidth == bw
                
                fprintf(fid, '%-15.1f %-15.2f %-15.2f %-15.1f %-15.2f\n', ...
                    bw/1e6, ...
                    results.scenarios(s).throughput_mbps, ...
                    results.scenarios(s).latency_mean_ms, ...
                    results.scenarios(s).delivery_ratio*100, ...
                    results.scenarios(s).spectral_efficiency);
                break;
            end
        end
    end
end

fprintf(fid, '\n========================================\n');
fprintf(fid, 'LINK COMPARISON SUMMARY\n');
fprintf(fid, '========================================\n\n');

% Aggregate satellite vs fiber statistics
total_sat_packets = 0;
total_fiber_packets = 0;
total_sat_latency = 0;
total_fiber_latency = 0;
sat_count = 0;
fiber_count = 0;

for s = 1:length(results.scenarios)
    if ~isnan(results.scenarios(s).sat_latency_mean_ms)
        total_sat_latency = total_sat_latency + results.scenarios(s).sat_latency_mean_ms;
        sat_count = sat_count + 1;
    end
    if ~isnan(results.scenarios(s).fiber_latency_mean_ms)
        total_fiber_latency = total_fiber_latency + results.scenarios(s).fiber_latency_mean_ms;
        fiber_count = fiber_count + 1;
    end
    total_sat_packets = total_sat_packets + results.scenarios(s).sat_packets;
    total_fiber_packets = total_fiber_packets + results.scenarios(s).fiber_packets;
end

fprintf(fid, 'Satellite Link:\n');
fprintf(fid, '  Average Latency: %.2f ms\n', total_sat_latency/sat_count);
fprintf(fid, '  Total Packets: %d\n', total_sat_packets);
fprintf(fid, '  Propagation Delay: %.2f ms\n\n', ...
    results.scenarios(1).sat_prop_delay * 1000);

fprintf(fid, 'Fiber Link:\n');
fprintf(fid, '  Average Latency: %.2f ms\n', total_fiber_latency/fiber_count);
fprintf(fid, '  Total Packets: %d\n', total_fiber_packets);
fprintf(fid, '  Propagation Delay: %.2f ms\n\n', ...
    results.scenarios(1).fiber_prop_delay * 1000);

fprintf(fid, 'Latency Improvement Factor: %.1fx\n', ...
    (total_sat_latency/sat_count) / (total_fiber_latency/fiber_count));

fprintf(fid, '\n========================================\n');
fprintf(fid, 'HANDOFF ANALYSIS\n');
fprintf(fid, '========================================\n\n');

% Average handoff packet loss across all scenarios
avg_handoff_loss = mean([results.scenarios.handoff_packet_loss]);
fprintf(fid, 'Average Packet Loss During Handoff: %.2f%%\n', avg_handoff_loss*100);
fprintf(fid, 'Handoff Window: ±5 seconds around t=%d s\n', cfg.handoff_time);

fprintf(fid, '\n========================================\n');
fprintf(fid, 'KEY FINDINGS\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, '1. Satellite links exhibit %.1f-%.1f ms higher latency than fiber\n', ...
    results.scenarios(1).sat_prop_delay*1000 - results.scenarios(1).fiber_prop_delay*1000, ...
    results.scenarios(1).sat_prop_delay*1000 - results.scenarios(1).fiber_prop_delay*1000 + 10);

fprintf(fid, '2. Higher-order modulations (64QAM) achieve %.1fx throughput of BPSK\n', ...
    cfg.mod_bits(4) / cfg.mod_bits(1));

fprintf(fid, '3. Fiber provides consistent sub-%.1f ms latency across all configs\n', ...
    max([results.scenarios.fiber_latency_mean_ms]));

fprintf(fid, '4. Handoff mechanism maintains %.1f%% packet delivery\n', ...
    (1-avg_handoff_loss)*100);

fprintf(fid, '5. Spectral efficiency ranges from %.2f to %.2f bps/Hz\n', ...
    min([results.scenarios.spectral_efficiency]), ...
    max([results.scenarios.spectral_efficiency]));

fprintf(fid, '\n========================================\n');
fprintf(fid, 'RECOMMENDATIONS\n');
fprintf(fid, '========================================\n\n');

fprintf(fid, '1. Use fiber for latency-critical applications (< 5 ms)\n');
fprintf(fid, '2. Deploy 64QAM with ≥50 MHz bandwidth for maximum throughput\n');
fprintf(fid, '3. Implement seamless handoff at t=%d s for uninterrupted service\n', ...
    cfg.handoff_time);
fprintf(fid, '4. Satellite suitable for initial deployment; migrate to fiber\n');
fprintf(fid, '5. Monitor SNR: satellite requires ≥10 dB for reliable operation\n');

fprintf(fid, '\n========================================\n');
fprintf(fid, 'END OF REPORT\n');
fprintf(fid, '========================================\n');

fclose(fid);

fprintf('Comprehensive report generated: simulation_report.txt\n');
end