%% ============================================================
% NOMP for 3D (AoA, Delay, Doppler) with Rx ULA, Tx=1
% - Input:  H (Nrx x M x Ns)
% - Output: est struct (phi, tau, nu, gamma, indices, residual)
% Author: (your name)
% ============================================================

clear; clc;

%% ========= USER PARAMETERS (EDIT HERE) ======================
load('h_freq.mat')
[~,~,Nrx,~,Ntx,Ns,M]=size(h_freq)
H = zeros(Nrx,M,Ns);
H = squeeze(h_freq(:,:,:,:,1,:,:));
H = permute(H,[1 3 2])

% [~,~,Nrx,~,Ntx,Ns,M]=size(h_freq)

% Array / signal sizes
% Nrx   = 64;         % Rx antennas
% M     = 56;          % # of used subcarriers (e.g., 6 DM-RS tones in the RB)
% Ns    = 14;         % # of OFDM symbols observed
Nx = sqrt(Nrx);
Ny = sqrt(Nrx);
K_max = 13;          % Max # of paths to extract

% Carrier / OFDM
f0    = 120e3;       % Subcarrier spacing [Hz] (e.g., 15 kHz; change if needed)
fc    = 10e9;       % Carrier frequency [Hz]
c     = 3e8;        % Speed of light [m/s]
lambda = c/fc;      % Wavelength [m]
d     = lambda/2;   % ULA inter-element spacing [m] (half-wavelength)
cp_len = 6;
Fs = f0 * 1024;
Tsym  = 1/f0 + cp_len/Fs;       % OFDM symbol duration (CP ignored; adjust if needed)

% Oversampling factors for initial grids (higher -> finer grid, heavier compute)
ov_phi = 4;         % AoA grid oversampling (~ Nrx*ov_phi points)
ov_tau = 4;         % Delay grid oversampling (~ M*ov_tau points)
ov_nu  = 4;         % Doppler grid oversampling (~ Ns*ov_nu points)

% Bounds (delay, Doppler)
tau_max = 1/f0;                 % Delay unambiguous range
nu_min  = -1/(2*Tsym);          % Doppler min [Hz]
nu_max  = +1/(2*Tsym);          % Doppler max [Hz]

% Newton refinement
newton_iters = 5;   % # coordinate-cycling Newton steps per selected path

% Frequency-domain shaping filter response g_m (length M). If unknown -> ones.
g_m = ones(M,1);

% Stopping criterion (optional): stop if residual power ratio < thr
use_stop      = true;
resid_thr_rel = 1e-3;   % stop if (||R||^2 / ||Y||^2) < this

% === Add noise according to Ptx, N0 (per complex dim) ===


% sigma2_H = N0 / Ptx;
SNR_dB = 20;
Ptx    = 1;                        % Ptx: (|X|^2) per subcarrier-symbol
N0     = Ptx / (10^(SNR_dB/10));   % N0 : (per complex dim)

% H_obs = add_estimation_noise_to_H(H, Ptx, N0);

% ==== Provide your observed channel H here ====
% H must be loaded or assigned as size (Nrx x M x Ns).
% Example (REMOVE in real use): make a dummy H if not present
if ~exist('H','var')
    warning('No H in workspace. Creating a toy H with 2 paths for a smoke test.');
    % Toy scene: 2 paths
    true_phi = [10, -20]*pi/180;
    true_tau = [2.1e-5, 7.0e-5];
    true_nu  = [100, -150];
    gamma    = [1.0*exp(1j*0.3), 0.6*exp(-1j*1.2)];
    H = synth_toy_channel(Nrx,M,Ns,true_phi,true_tau,true_nu,gamma,f0,Tsym,d/lambda);
end

%% ========= DERIVED INDICES =================================
m_idx = (0:M-1).';     % subcarrier indices
s_idx = (0:Ns-1).';    % OFDM symbol indices
n_vec = (0:Nrx-1).';

%% ========= BUILD INITIAL GRIDS & DICTIONARIES ===============
% AoA grid (sin-uniform)
Nxg = Nx * ov_phi;                   % grid density along x
Nyg = Ny * ov_phi;                   % grid density along y
ux_grid = linspace(-1, 1, Nxg);      % direction cosines along x
uy_grid = linspace(-1, 1, Nyg);      % direction cosines along y
nx = (0:Nx-1).'; ny = (0:Ny-1).';
Ax = exp(-1j*2*pi*(d/lambda) * nx * ux_grid);   % (Nx x Nxg)
Ay = exp(-1j*2*pi*(d/lambda) * ny * uy_grid);   % (Ny x Nyg)

% A_phi = kron(Ax, Ay);                            % (Nrx x (Nxg*Nyg))  ← 이름 유지
% Nphi  = size(A_phi,2);
[UX, UY] = meshgrid(ux_grid, uy_grid);   % (Nyg x Nxg)
diskMask = (UX.^2 + UY.^2) <= 1;          % keep only physical directions (front hemisphere)

% Build full kron dictionary then select masked columns
A_full   = kron(Ax, Ay);                  % (Nrx x (Nxg*Nyg)) with x-fast stacking
ux_cols_full = UX(:);  uy_cols_full = UY(:);
keep_idx = find(diskMask(:));
A_phi    = A_full(:, keep_idx);           % (Nrx x Nphi)  ← 이름 유지
ux_cols  = ux_cols_full(keep_idx);        % (Nphi x 1)
uy_cols  = uy_cols_full(keep_idx);
Nphi     = size(A_phi,2);
% ux_cols  = UX(:);                        % (Nphi x 1)
% uy_cols  = UY(:);

% azimuth (horizontal)
phi_hor_cols = atan2(uy_cols, ux_cols);   % [-pi, pi]
% elevation from horizon (0° at horizon, 90° at broadside +z)
uz_cols      = sqrt(max(0, 1 - ux_cols.^2 - uy_cols.^2));
phi_ver_cols = asin(uz_cols);             % [0, pi/2]

% Delay grid
Ntau = M * ov_tau;
tau_grid = linspace(0, tau_max, Ntau);                       % [s]
A_tau = exp(-1j*2*pi * (m_idx) * f0 .* tau_grid);            % (M x Ntau)

% Doppler grid
Nnu = Ns * ov_nu;
nu_grid = linspace(nu_min, nu_max, Nnu);                     % [Hz]
A_nu = exp( 1j*2*pi * (s_idx*Tsym) .* nu_grid );             % (Ns x Nnu)

% Column norms (for normalized correlation)
colnorm_phi = sqrt(sum(abs(A_phi).^2,1)).';   % (Nphi x 1)
colnorm_tau = sqrt(sum(abs(A_tau).^2,1)).';   % (Ntau x 1)
colnorm_nu  = sqrt(sum(abs(A_nu ).^2,1)).';   % (Nnu  x 1)

%% ========= PREPARE OBSERVATION ==============================
% Frequency-axis compensation (divide by g_m)
gm  = reshape(g_m(:).', [1 M 1]);       % (1 x M x 1)
Y   = H ./ gm;                           % compensated observation
E0  = sum(abs(Y(:)).^2);                 % initial energy

%% ========= MAIN NOMP LOOP ===================================
est.idx_phi = zeros(K_max,1);
est.idx_tau = zeros(K_max,1);
est.idx_nu  = zeros(K_max,1);
est.phi     = zeros(K_max,1);
est.tau     = zeros(K_max,1);
est.nu      = zeros(K_max,1);
est.gamma   = zeros(K_max,1);

R = Y;  % residual
for k = 1:K_max
    % --- 3D correlation map via successive projections ---
    Z1 = proj_along_rx(R, A_phi);          % (Nphi x M x Ns)
    Z2 = proj_along_delay(Z1, A_tau);      % (Nphi x Ntau x Ns)
    C  = proj_along_doppler(Z2, A_nu);     % (Nphi x Ntau x Nnu)

    % normalized correlation
    denom = reshape(colnorm_phi,[Nphi 1 1]) .* ...
            reshape(colnorm_tau,[1 Ntau 1]) .* ...
            reshape(colnorm_nu ,[1 1 Nnu ]);
    Cn = abs(C) ./ denom;

    [~, lin_idx] = max(Cn(:));
    [i_phi, i_tau, i_nu] = ind2sub(size(Cn), lin_idx);

    % grid initial guesses
    ux0 = ux_cols(i_phi);  uy0 = uy_cols(i_phi);
    tau0 = tau_grid(i_tau); nu0 = nu_grid(i_nu);

    % --- Newton refinement (coordinate-cycling) ---
    [ux_r, uy_r, tau_r, nu_r, alpha_r, R] = refine_upa_newton( ...
    R, ux0, uy0, tau0, nu0, ...
    f0, Tsym, d/lambda, Nx, Ny, m_idx, s_idx, ...
    newton_iters, [0 tau_max], [nu_min nu_max]);

    % store
    est.idx_phi(k)  = i_phi;
    est.idx_tau(k)  = i_tau;
    est.idx_nu(k)   = i_nu;
    est.ux(k)       = ux_r;
    est.uy(k)       = uy_r;
    est.phi_hor(k)  = phi_hor_cols(i_phi);   % [rad]
    est.phi_ver(k)  = phi_ver_cols(i_phi);   % [rad]
    est.tau(k)      = tau_r;
    est.nu(k)       = nu_r;
    est.gamma(k)    = alpha_r;
    est.theta_off(k) = asin( sqrt(max(0, ux_r^2 + uy_r^2)) );

    % stopping (optional)
    if use_stop
        if sum(abs(R(:)).^2)/E0 < resid_thr_rel
            est = trim_est(est, k);  % keep first k entries
            break;
        end
    end
end

% ---- (Optional) Debias: joint LS re-estimation of gains on refined atoms
[Gamma_ls, R_final] = debias_joint_ls_UPA(R + 0, Y, est, Nx, Ny, M, Ns, f0, Tsym, d/lambda, m_idx, s_idx);
est.gamma = Gamma_ls;
est.R     = R_final;

%% ========= REPORT ===========================================
vel_mps = est.nu * lambda;  % 도플러→속도
T = table((1:numel(est.tau)).', ...
          rad2deg(est.phi_hor(:)), rad2deg(est.phi_ver(:)), ...
          est.tau(:), vel_mps(:), est.gamma(:), ...
   'VariableNames', {'path','ang_hor_deg','ang_ver_deg','tau_s','vel_mps','gamma'});
disp(T);


H_est = reconstruct_H_from_est_UPA(est, Nx, Ny, M, Ns, f0, Tsym, d, lambda, g_m);
fprintf('NMSE = %.2f dB\n', nmse_db(H, H_est));


opts = struct('tau_unit','us','title','DD scatter (v in m/s)');
plot_delay_doppler_scatter([], [], [], ...             % true가 없으면 빈 벡터
                           est.tau, est.nu*lambda, est.gamma, opts);
ylabel('v [m/s]');


figure();
plot_dd_heatmap_from_H(H, f0, Tsym, true);      % true/est 각각 호출해서 비교
figure();
plot_dd_heatmap_from_H(H_est, f0, Tsym, true);


%plot_param_scatter(true_phi, true_tau, true_nu, est.phi, est.tau, est.nu);
% Parameter comparison

%% ========= LOCAL FUNCTIONS =================================

function Z1 = proj_along_rx(R, A_phi)
    % R: (Nrx x M x Ns), A_phi: (Nrx x Nphi)
    [Nrx, M, Ns] = size(R);
    R2 = reshape(R, [Nrx, M*Ns]);
    Z1 = A_phi' * R2;                     % (Nphi x M*Ns)
    Z1 = reshape(Z1, [size(A_phi,2), M, Ns]);
end

function Z2 = proj_along_delay(Z1, A_tau)
    % Z1: (Nphi x M x Ns), A_tau: (M x Ntau)
    [Nphi, ~, Ns] = size(Z1);
    Ntau = size(A_tau,2);
    Z2 = zeros(Nphi, Ntau, Ns, 'like', Z1);
    AtH = conj(A_tau);  % (M x Ntau)
    for s = 1:Ns
        Zs = Z1(:,:,s);                 % (Nphi x M)
        Z2(:,:,s) = Zs * AtH;           % (Nphi x Ntau)
    end
end

function C = proj_along_doppler(Z2, A_nu)
    % Z2: (Nphi x Ntau x Ns), A_nu: (Ns x Nnu)
    [Nphi, Ntau, Ns] = size(Z2);
    AnH = conj(A_nu);             % (Ns x Nnu)
    C = zeros(Nphi, Ntau, size(A_nu,2), 'like', Z2);
    for ip = 1:Nphi
        Zp = squeeze(Z2(ip,:,:)); % (Ntau x Ns)
        C(ip,:,:) = Zp * AnH;     % (Ntau x Nnu)
    end
end

function [phi, tau, nu, alpha, R] = refine_atom_newton( ...
        R, phi, tau, nu, f0, Tsym, d_over_lambda, m_idx, s_idx, max_iter, tau_bounds, nu_bounds)

    [Nrx, M, Ns] = size(R);
    n_vec = (0:Nrx-1).';

    for it = 1:max_iter
        % build atoms & derivatives (current)
        [aphi, atau, anu, dphi, d2phi, dtau, d2tau, dnu, d2nu] = build_atoms(phi, tau, nu);

        % --- phi update ---
        [z, z1, z2] = corr_and_deriv_phi(R, aphi, atau, anu, dphi, d2phi);
        df  = 2*real(conj(z).*z1);
        d2f = 2*( real(conj(z).*z2) + abs(z1).^2 );
        step = - df / (d2f + eps);
        phi  = max(min(phi + step, +pi/2-1e-6), -pi/2+1e-6);

        % rebuild with new phi
        [aphi, atau, anu, dphi, d2phi, dtau, d2tau, dnu, d2nu] = build_atoms(phi, tau, nu);

        % --- tau update ---
        [z, z1, z2] = corr_and_deriv_tau(R, aphi, atau, anu, dtau, d2tau);
        df  = 2*real(conj(z).*z1);
        d2f = 2*( real(conj(z).*z2) + abs(z1).^2 );
        step = - df / (d2f + eps);
        tau  = min(max(tau + step, tau_bounds(1)), tau_bounds(2));

        % rebuild with new tau
        [aphi, atau, anu, dphi, d2phi, dtau, d2tau, dnu, d2nu] = build_atoms(phi, tau, nu);

        % --- nu update ---
        [z, z1, z2] = corr_and_deriv_nu(R, aphi, atau, anu, dnu, d2nu);
        df  = 2*real(conj(z).*z1);
        d2f = 2*( real(conj(z).*z2) + abs(z1).^2 );
        step = - df / (d2f + eps);
        nu   = min(max(nu + step, nu_bounds(1)), nu_bounds(2));
    end

    % final LS gain and residual update
    [aphi, atau, anu] = build_atoms(phi, tau, nu);
    atom_norm2 = (Nrx) * (M) * (Ns);  % all entries have |.|=1
    % z = <R, a>
    z = 0;
    A_nm = aphi * atau.';
    for s = 1:Ns
        Rs = R(:,:,s);
        z  = z + sum(conj(A_nm(:)).*Rs(:)) * conj(anu(s));
    end
    alpha = z / atom_norm2;
    % subtract from residual
    for s = 1:Ns
        R(:,:,s) = R(:,:,s) - alpha * A_nm * anu(s);
    end

    % nested helpers
    function [aphi, atau, anu, dphi, d2phi, dtau, d2tau, dnu, d2nu] = build_atoms(phi, tau, nu)
        % AoA
        phase_phi = -1j*2*pi*d_over_lambda * n_vec * sin(phi);
        aphi = exp(phase_phi);  % (Nrx x 1)
        c_n  = (-1j*2*pi*d_over_lambda * n_vec * cos(phi));
        dphi = aphi .* c_n;
        c2_n = (c_n.^2 + (1j*2*pi*d_over_lambda * n_vec * sin(phi)));
        d2phi = aphi .* c2_n;

        % Delay
        a_tau_phase = -1j*2*pi*f0*tau*m_idx;
        atau = exp(a_tau_phase);
        k_m  = (-1j*2*pi*f0*m_idx);
        dtau  = atau .* k_m;
        d2tau = atau .* (k_m.^2);

        % Doppler
        a_nu_phase = 1j*2*pi*nu*Tsym*s_idx;
        anu = exp(a_nu_phase);
        k_s  = (1j*2*pi*Tsym*s_idx);
        dnu  = anu .* k_s;
        d2nu = anu .* (k_s.^2);
    end

    function [z, z1, z2] = corr_and_deriv_phi(R, aphi, atau, anu, dphi, d2phi)
        [~, ~, Ns] = size(R);
        v = zeros(size(aphi), 'like', R);
        for s = 1:Ns
            v = v + ( R(:,:,s) * conj(atau) ) * conj(anu(s));  % (Nrx x 1)
        end
        z  = aphi'  * v;
        z1 = dphi'  * v;
        z2 = d2phi' * v;
    end

    function [z, z1, z2] = corr_and_deriv_tau(R, aphi, atau, anu, dtau, d2tau)
        [~, M, Ns] = size(R);
        u = zeros(M,1,'like',R);
        for s = 1:Ns
            u = u + ((conj(aphi).') * R(:,:,s)).' * conj(anu(s)); % (M x 1)
        end
        z  = (conj(atau))'  * u;
        z1 = (conj(dtau))'  * u;
        z2 = (conj(d2tau))' * u;
    end

    function [z, z1, z2] = corr_and_deriv_nu(R, aphi, atau, anu, dnu, d2nu)
        [~, ~, Ns] = size(R);
        A_nm = conj(aphi) * conj(atau)';  % (Nrx x M)
        w = zeros(Ns,1,'like',R);
        for s = 1:Ns
            Rs = R(:,:,s);
            w(s) = sum( conj(A_nm(:)) .* Rs(:) );
        end
        z  = (conj(anu))'  * w;
        z1 = (conj(dnu))'  * w;
        z2 = (conj(d2nu))' * w;
    end
end

function [Gamma_ls, R_out] = debias_joint_ls(R_curr, Y, est, Nrx, M, Ns, f0, Tsym, d_over_lambda, n_vec, m_idx, s_idx)
    % Build design matrix for all refined atoms and LS re-estimate gains.
    K = numel(est.phi);
    if K==0
        Gamma_ls = [];
        R_out = Y;
        return;
    end
    % Each atom -> vectorized (Nrx*M*Ns x 1)
    A = zeros(Nrx*M*Ns, K, 'like', Y);
    for k = 1:K
        [aphi, atau, anu] = build_atoms_no_deriv(est.phi(k), est.tau(k), est.nu(k), f0, Tsym, d_over_lambda, n_vec, m_idx, s_idx);
        A_nm = aphi * atau.';
        col = zeros(Nrx*M*Ns,1,'like',Y);
        idx = 1;
        for s = 1:Ns
            col((s-1)*Nrx*M+(1:Nrx*M)) = A_nm(:) * anu(s);
        end
        A(:,k) = col;
    end
    y = reshape(Y, [], 1);
    Gamma_ls = A \ y;  % least-squares
    % residual
    R_vec = y - A*Gamma_ls;
    R_out = reshape(R_vec, [Nrx, M, Ns]);
end

function [aphi, atau, anu] = build_atoms_no_deriv(phi, tau, nu, f0, Tsym, d_over_lambda, n_vec, m_idx, s_idx)
    aphi = exp(-1j*2*pi*d_over_lambda * n_vec * sin(phi));
    atau = exp(-1j*2*pi*f0*tau*m_idx);
    anu  = exp( 1j*2*pi*nu*Tsym*s_idx);
end

function est = trim_est(est, k)
    flds = fieldnames(est);
    for i = 1:numel(flds)
        v = est.(flds{i});
        if isnumeric(v) && size(v,1) >= k
            est.(flds{i}) = v(1:k,:);
        end
    end
end

function H = synth_toy_channel(Nrx,M,Ns,phi,tau,nu,gamma,f0,Tsym,d_over_lambda)
    % Simple channel synthesizer for a few paths (Tx=1)
    K = numel(phi);
    n_vec = (0:Nrx-1).';
    m_idx = (0:M-1).';
    s_idx = (0:Ns-1).';
    H = zeros(Nrx,M,Ns);
    for k = 1:K
        aphi = exp(-1j*2*pi*d_over_lambda * n_vec * sin(phi(k)));
        atau = exp(-1j*2*pi*f0*tau(k)*m_idx);
        anu  = exp( 1j*2*pi*nu(k)*Tsym*s_idx);
        A_nm = aphi * atau.';
        for s = 1:Ns
            H(:,:,s) = H(:,:,s) + gamma(k) * A_nm * anu(s);
        end
    end
    % add a touch of noise
    H = H + 0*(randn(size(H))+1j*randn(size(H)))*1e-3;
end


function H_est = reconstruct_H_from_est(est, Nrx, M, Ns, f0, Tsym, d, lambda, g_m)
    % est: (refined) est.phi, est.tau, est.nu, est.gamma
    % g_m: (M x 1)  frequency shaping; if ones(M,1), 그대로
    d_over_lambda = d / lambda;
    n_vec = (0:Nrx-1).';
    m_idx = (0:M-1).';
    s_idx = (0:Ns-1).';

    H_est = zeros(Nrx, M, Ns);
    for k = 1:numel(est.phi)
        aphi = exp(-1j*2*pi*d_over_lambda * n_vec * sin(est.phi(k)));
        atau = exp(-1j*2*pi*f0*est.tau(k)*m_idx);
        anu  = exp( 1j*2*pi*est.nu(k)*Tsym*s_idx);
        A_nm = aphi * atau.';
        for s = 1:Ns
            H_est(:,:,s) = H_est(:,:,s) + est.gamma(k) * A_nm * anu(s);
        end
    end

    % 주파수 축 복원: 처음에 Y = H./g_m로 나눴었으므로 여기서 다시 곱해줌
    gm = reshape(g_m(:).', [1 M 1]);      % (1 x M x 1)
    H_est = H_est .* gm;
end


function nmse = nmse_db(H_true, H_est)
    nmse_lin = sum(abs(H_true(:) - H_est(:)).^2) / sum(abs(H_true(:)).^2);
    nmse     = 10*log10(nmse_lin);
end


function [perm] = match_params(true_phi, true_tau, true_nu, est_phi, est_tau, est_nu)
    % 간단한 최근접(헝가리안 없이) 매칭: K가 작을 때 충분
    Kt = numel(true_phi); Ke = numel(est_phi);
    K = min(Kt, Ke);
    % 정규화 스케일
    s_phi = 1;                        % 라디안 그대로
    s_tau = max(std(true_tau)+eps, 1e-9);
    s_nu  = max(std(true_nu )+eps, 1e-9);

    used = false(Ke,1);
    perm = zeros(K,1);
    for i = 1:Kt
        cost = ((est_phi-true_phi(i))/s_phi).^2 + ...
               ((est_tau-true_tau(i))/s_tau).^2 + ...
               ((est_nu -true_nu (i))/s_nu ).^2;
        cost(used) = inf;
        [~, j] = min(cost);
        if j<=Ke
            perm(i) = j; used(j) = true;
        end
    end
end

function plot_param_scatter(true_phi, true_tau, true_nu, est_phi, est_tau, est_nu)
    Kt = numel(true_phi); Ke = numel(est_phi);
    K = min(Kt, Ke);
    perm = match_params(true_phi, true_tau, true_nu, est_phi, est_tau, est_nu);

    figure; 
    subplot(3,1,1);
    plot((180/pi)*true_phi(1:K),'o'); hold on;
    plot((180/pi)*est_phi(perm(1:K)),'x');
    ylabel('\phi [deg]'); legend('true','est'); grid on;

    subplot(3,1,2);
    plot(true_tau(1:K)*1e6,'o'); hold on;
    plot(est_tau(perm(1:K))*1e6,'x');
    ylabel('\tau [\mus]'); legend('true','est'); grid on;

    subplot(3,1,3);
    plot(true_nu(1:K),'o'); hold on;
    plot(est_nu(perm(1:K)),'x');
    ylabel('\nu [Hz]'); xlabel('path index'); legend('true','est'); grid on;
end


function plot_delay_doppler_scatter(true_tau, true_nu, true_gamma, ...
                                    est_tau,  est_nu,  est_gamma,  opts)
% true_* / est_*: 벡터 (길이는 다 달라도 OK)
%  - tau [s], nu [Hz], gamma 복소이득 (|gamma|는 마커 크기, ∠gamma는 색상)
% opts (struct, optional):
%  .tau_unit  = 'us' or 'ns' or 's' (default 'us')
%  .size_min  = 40;  .size_max = 160;    % 마커 크기 범위
%  .title     = 'Delay-Doppler Scatter';

    if nargin < 7 || isempty(opts), opts = struct(); end
    if ~isfield(opts,'tau_unit'),  opts.tau_unit = 'us'; end
    if ~isfield(opts,'size_min'),  opts.size_min = 40; end
    if ~isfield(opts,'size_max'),  opts.size_max = 160; end
    if ~isfield(opts,'title'),     opts.title    = 'Delay–Doppler Scatter'; end

    % --- 단위 변환
    switch lower(opts.tau_unit)
        case 'us', tau_scale = 1e6; tau_label = '\tau [\mus]';
        case 'ns', tau_scale = 1e9; tau_label = '\tau [ns]';
        otherwise, tau_scale = 1;   tau_label = '\tau [s]';
    end

    % --- 크기 스케일 (진폭 → 마커 size)
    function sz = mags2sizes(g)
        mag = abs(g(:));
        if isempty(mag), sz = []; return; end
        m1 = min(mag); m2 = max(mag);
        if m2>m1
            w = (mag - m1) / (m2 - m1);
        else
            w = ones(size(mag))*0.6;
        end
        sz = opts.size_min + w*(opts.size_max-opts.size_min);
    end

    % --- 위상 → 색상 (∠gamma in [-pi,pi] → colormap)
    function col = phase2color(g)
        ph = angle(g(:));
        if isempty(ph), col = []; return; end
        % HSV 색상환을 사용해도 되지만, 간단히 jet로 매핑
        % 정규화 [0,1]
        t = (ph + pi) / (2*pi);
        cmap = jet(256);
        idx = max(1, min(256, round(1 + t*255)));
        col = cmap(idx, :);
    end

    tau_true = true_tau(:)*tau_scale;  nu_true = true_nu(:);
    tau_est  = est_tau(:) *tau_scale;  nu_est  = est_nu(:);
    sz_true  = mags2sizes(true_gamma);
    sz_est   = mags2sizes(est_gamma);
    col_true = phase2color(true_gamma);
    col_est  = phase2color(est_gamma);

    figure; hold on; box on; grid on;
    % Estimated 먼저(밑에) 찍고, True를 위에 찍어 시각적 구분
    if ~isempty(tau_est)
        s1 = scatter(tau_est, nu_est, sz_est, col_est, 'filled', 'MarkerEdgeColor','k','MarkerFaceAlpha',0.75);
    end
    if ~isempty(tau_true)
        s2 = scatter(tau_true, nu_true, sz_true, col_true, 'filled', 'MarkerEdgeColor','w', 'LineWidth',1.0);
    end

    xlabel(tau_label); ylabel('\nu [Hz]');
    title(opts.title);
    % 범례
    if ~isempty(tau_true) && ~isempty(tau_est)
        legend({'Estimated','True'}, 'Location','best'); 
    elseif ~isempty(tau_true)
        legend({'True'}, 'Location','best');
    else
        legend({'Estimated'}, 'Location','best');
    end
    % 색상바(추정 위상 기준)
    cb = colorbar; cb.Label.String = 'Phase \angle\gamma [rad]';
    colormap(jet);
    axis tight;
end

function plot_dd_heatmap_from_H(H, f0, Tsym, sum_over_rx)
% H: (Nrx x M x Ns)
% sum_over_rx=true 면 Rx 축 합산 후 2D-DFT; false면 첫 안테나만 사용
    if nargin<4, sum_over_rx=true; end
    [Nrx,M,Ns] = size(H);
    if sum_over_rx
        H2 = squeeze(sum(H,1));    % (M x Ns)
    else
        H2 = squeeze(H(1,:,:));    % (M x Ns)
    end
    % 2D-DFT (m->tau, s->nu)
    S = fftshift(fft2(H2),1);
    S = fftshift(S,2);
    tau_axis = ((-floor(M/2):ceil(M/2)-1) / (M*f0));     % [s]
    nu_axis  = ((-floor(Ns/2):ceil(Ns/2)-1) / (Ns*Tsym));% [Hz]
    imagesc(tau_axis*1e6, nu_axis, 20*log10(abs(S.'/max(abs(S(:))+eps))));
    axis xy; xlabel('\tau [\mus]'); ylabel('\nu [Hz]'); colorbar;
    title('Delay–Doppler heatmap');
end

% 기존 시뮬 레벨에서 노이즈를 아주 작게 더하던 부분을 아래로 교체:



function H_obs = add_estimation_noise_to_H(H_true, Ptx, N0)
    sigma2_H = N0 / max(Ptx, eps);
    sigma_H  = sqrt(sigma2_H/2);
    W = sigma_H*(randn(size(H_true)) + 1j*randn(size(H_true)));
    H_obs = H_true + W;
end
function a = upa_steer_vec(Nx,Ny,d_over_lambda, ux, uy)
    nx = (0:Nx-1).'; ny = (0:Ny-1).';
    ax = exp(-1j*2*pi*d_over_lambda * nx * ux);  % (Nx x 1)
    ay = exp(-1j*2*pi*d_over_lambda * ny * uy);  % (Ny x 1)
    a  = kron(ax, ay);                            % (Nx*Ny x 1)
end

function [tau, nu, alpha, R] = refine_tau_nu_given_aphi( ...
        R, aphi, tau, nu, f0, Tsym, m_idx, s_idx, max_iter, tau_bounds, nu_bounds)

    [Nrx,M,Ns] = size(R);

    for it = 1:max_iter
        % ----- delay derivatives -----
        atau = exp(-1j*2*pi*f0*tau*m_idx);
        dtau = atau .* (-1j*2*pi*f0*m_idx);
        d2tau= atau .* ((-1j*2*pi*f0*m_idx).^2);

        % collapse over Rx for delay update
        u = zeros(M,1,'like',R);
        for s=1:Ns
            u = u + ((conj(aphi).')*R(:,:,s)).' * 1;   % later multiply by conj(anu) when nu updated
        end
        % Doppler is not yet applied in u; use anu=1 for this step
        z  = (conj(atau))'  * u;
        z1 = (conj(dtau))'  * u;
        z2 = (conj(d2tau))' * u;
        df  = 2*real(conj(z).*z1);
        d2f = 2*( real(conj(z).*z2) + abs(z1).^2 );
        tau = min(max(tau - df/(d2f + eps), tau_bounds(1)), tau_bounds(2));

        % ----- doppler derivatives -----
        anu = exp( 1j*2*pi*nu*Tsym*s_idx);
        dnu  = anu .* ( 1j*2*pi*Tsym*s_idx);
        d2nu = anu .* ((1j*2*pi*Tsym*s_idx).^2);

        % collapse to Ns for doppler update
        A_nm = conj(aphi) * conj(atau)';  % (Nrx x M)
        w = zeros(Ns,1,'like',R);
        for s=1:Ns
            Rs = R(:,:,s);
            w(s) = sum( conj(A_nm(:)) .* Rs(:) );
        end
        z  = (conj(anu))'  * w;
        z1 = (conj(dnu))'  * w;
        z2 = (conj(d2nu))' * w;
        df  = 2*real(conj(z).*z1);
        d2f = 2*( real(conj(z).*z2) + abs(z1).^2 );
        nu  = min(max(nu - df/(d2f + eps), nu_bounds(1)), nu_bounds(2));
    end

    % final LS amplitude & residual update
    atau = exp(-1j*2*pi*f0*tau*m_idx);
    anu  = exp( 1j*2*pi*nu*Tsym*s_idx);
    atom_norm2 = (Nrx) * (M) * (Ns);
    z = 0;
    A_nm = aphi * atau.';
    for s=1:Ns
        Rs = R(:,:,s);
        z  = z + sum(conj(A_nm(:)).*Rs(:)) * conj(anu(s));
    end
    alpha = z / atom_norm2;
    for s=1:Ns
        R(:,:,s) = R(:,:,s) - alpha * A_nm * anu(s);
    end
end
function [Gamma_ls, R_out] = debias_joint_ls_UPA(R_curr, Y, est, Nx, Ny, M, Ns, f0, Tsym, d_over_lambda, m_idx, s_idx)
    K = numel(est.tau);
    Nrx = Nx*Ny;
    if K==0, Gamma_ls=[]; R_out=Y; return; end
    A = zeros(Nrx*M*Ns, K, 'like', Y);
    for k=1:K
        aphi = upa_steer_vec(Nx,Ny,d_over_lambda, est.ux(k), est.uy(k));
        atau = exp(-1j*2*pi*f0*est.tau(k)*m_idx);
        anu  = exp( 1j*2*pi*est.nu(k)*Tsym*s_idx);
        A_nm = aphi * atau.';
        col = zeros(Nrx*M*Ns,1,'like',Y);
        for s=1:Ns
            col((s-1)*Nrx*M+(1:Nrx*M)) = A_nm(:) * anu(s);
        end
        A(:,k) = col;
    end
    y = reshape(Y, [], 1);
    Gamma_ls = A \ y;
    R_vec = y - A*Gamma_ls;
    R_out = reshape(R_vec, [Nrx, M, Ns]);
end
function H_est = reconstruct_H_from_est_UPA(est, Nx, Ny, M, Ns, f0, Tsym, d, lambda, g_m)
    d_over_lambda = d / lambda;
    m_idx = (0:M-1).';
    s_idx = (0:Ns-1).';
    Nrx = Nx*Ny;
    H_est = zeros(Nrx, M, Ns);
    for k = 1:numel(est.tau)
        aphi = upa_steer_vec(Nx,Ny,d_over_lambda, est.ux(k), est.uy(k));
        atau = exp(-1j*2*pi*f0*est.tau(k)*m_idx);
        anu  = exp( 1j*2*pi*est.nu(k)*Tsym*s_idx);
        A_nm = aphi * atau.';
        for s = 1:Ns
            H_est(:,:,s) = H_est(:,:,s) + est.gamma(k) * A_nm * anu(s);
        end
    end
    gm = reshape(g_m(:).', [1 M 1]);
    H_est = H_est .* gm;
end
function [ux, uy, tau, nu, alpha, R] = refine_upa_newton( ...
        R, ux, uy, tau, nu, f0, Tsym, d_over_lambda, Nx, Ny, m_idx, s_idx, ...
        max_iter, tau_bounds, nu_bounds)

    [Nrx,M,Ns] = size(R); %#ok<ASGLU> % Nrx = Nx*Ny
    nx = (0:Nx-1).'; ny = (0:Ny-1).';

    % 1D steering helpers
    ax_fun   = @(ux) exp(-1j*2*pi*d_over_lambda * nx * ux);
    ay_fun   = @(uy) exp(-1j*2*pi*d_over_lambda * ny * uy);
    dax_fun  = @(ax) ax .* (-1j*2*pi*d_over_lambda * nx);
    d2ax_fun = @(ax) ax .* ((-1j*2*pi*d_over_lambda * nx).^2);
    day_fun  = @(ay) ay .* (-1j*2*pi*d_over_lambda * ny);
    d2ay_fun = @(ay) ay .* ((-1j*2*pi*d_over_lambda * ny).^2);

    for it = 1:max_iter
        % ----- build atoms -----
        ax   = ax_fun(ux);  ay   = ay_fun(uy);
        dax  = dax_fun(ax); day  = day_fun(ay);
        d2ax = d2ax_fun(ax);d2ay = d2ay_fun(ay);

        aphi     = kron(ax, ay);           % (Nx*Ny x 1)
        d_aphi_x = kron(dax, ay);
        d_aphi_y = kron(ax, day);
        d2_aphi_x= kron(d2ax, ay);
        d2_aphi_y= kron(ax, d2ay);

        atau = exp(-1j*2*pi*f0*tau*m_idx);
        dtau = atau .* (-1j*2*pi*f0*m_idx);
        d2tau= atau .* ((-1j*2*pi*f0*m_idx).^2);

        anu  = exp( 1j*2*pi*nu*Tsym*s_idx);
        dnu  = anu  .* ( 1j*2*pi*Tsym*s_idx);
        d2nu = anu  .* ((1j*2*pi*Tsym*s_idx).^2);

        % ----- common collapsed vector v for angle updates -----
        v = zeros(Nx*Ny,1,'like',R);
        for s=1:Ns
            v = v + ( R(:,:,s) * conj(atau) ) * conj(anu(s)); % (Nrx x 1)
        end

        % ===== ux update =====
        z  = aphi'      * v;
        z1 = d_aphi_x'  * v;
        z2 = d2_aphi_x' * v;
        df  = 2*real(conj(z).*z1);
        d2f = 2*( real(conj(z).*z2) + abs(z1).^2 );
        ux  = ux - df/(d2f + eps);
        ux  = max(min(ux, 1), -1);

        % rebuild terms impacted by ux
        ax   = ax_fun(ux);  dax  = dax_fun(ax); d2ax = d2ax_fun(ax);
        aphi     = kron(ax, ay);
        d_aphi_x = kron(dax, ay);

        % ===== uy update =====
        z  = aphi'      * v;
        z1 = d_aphi_y'  * v;
        z2 = d2_aphi_y' * v;
        df  = 2*real(conj(z).*z1);
        d2f = 2*( real(conj(z).*z2) + abs(z1).^2 );
        uy  = uy - df/(d2f + eps);
        uy  = max(min(uy, 1), -1);

        % ===== tau update =====
        % collapse over Rx for delay update (anu applied later)
        u = zeros(M,1,'like',R);
        for s=1:Ns
            Rs = R(:,:,s);
            u = u + ((conj(aphi).') * Rs).' * conj(anu(s)); % (M x 1)
        end
        z  = (conj(atau))'  * u;
        z1 = (conj(dtau))'  * u;
        z2 = (conj(d2tau))' * u;
        df  = 2*real(conj(z).*z1);
        d2f = 2*( real(conj(z).*z2) + abs(z1).^2 );
        tau = min(max(tau - df/(d2f + eps), tau_bounds(1)), tau_bounds(2));

        % ===== nu update =====
        % collapse over Rx,M for doppler update with updated aphi,atau
        A_nm = conj(aphi) * conj(atau)';  % (Nrx x M)
        w = zeros(Ns,1,'like',R);
        for s=1:Ns
            Rs = R(:,:,s);
            w(s) = sum( conj(A_nm(:)) .* Rs(:) );
        end
        z  = (conj(anu))'  * w;
        z1 = (conj(dnu))'  * w;
        z2 = (conj(d2nu))' * w;
        df  = 2*real(conj(z).*z1);
        d2f = 2*( real(conj(z).*z2) + abs(z1).^2 );
        nu  = min(max(nu - df/(d2f + eps), nu_bounds(1)), nu_bounds(2));
    end

    % ----- final LS amplitude & residual update -----
    ax = ax_fun(ux); ay = ay_fun(uy);
    aphi = kron(ax, ay);
    atau = exp(-1j*2*pi*f0*tau*m_idx);
    anu  = exp( 1j*2*pi*nu*Tsym*s_idx);

    atom_norm2 = (Nx*Ny) * M * Ns;
    z = 0;
    A_nm = aphi * atau.';
    for s=1:Ns
        Rs = R(:,:,s);
        z = z + sum(conj(A_nm(:)).*Rs(:)) * conj(anu(s));
    end
    alpha = z / atom_norm2;

    for s=1:Ns
        R(:,:,s) = R(:,:,s) - alpha * A_nm * anu(s);
    end
end
