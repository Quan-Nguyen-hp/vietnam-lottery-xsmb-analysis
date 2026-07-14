import numpy as np
import pandas as pd
from datetime import datetime

class FeatureEngine:
    def __init__(self):
        pass

    def get_mirror(self, num: int) -> int:
        d1, d2 = num // 10, num % 10
        m1 = (d1 + 5) % 10
        m2 = (d2 + 5) % 10
        return m1 * 10 + m2

    def get_inverted(self, num: int) -> int:
        d1, d2 = num // 10, num % 10
        return d2 * 10 + d1

    def build_features_for_day(self, df_history: pd.DataFrame, S_history: np.ndarray, target_date: datetime) -> pd.DataFrame:
        """
        Build feature matrix of shape (100, num_features) for target_date.
        df_history contains lottery results before target_date.
        S_history is a binary matrix of shape (len(df_history), 100) representing appearances.
        """
        N = len(df_history)
        features = {num: {} for num in range(100)}

        # Basic configurations & targets
        target_weekday = target_date.weekday()
        target_day = target_date.day
        target_month = target_date.month

        # Calculate special prize digit sum for each day in history - Vectorized
        has_special = 'special' in df_history.columns
        specials_sum = np.zeros(N)
        if has_special and N > 0:
            specials_sum = df_history['special'].astype(str).str.zfill(2).apply(lambda s: sum(int(c) for c in s)).values

        # Precompute Head and Tail câm for history - Vectorized
        if N > 0:
            S_reshaped = S_history.reshape(N, 10, 10)
            head_cam_hist = (np.sum(S_reshaped, axis=2) == 0).astype(int)
            tail_cam_hist = (np.sum(S_reshaped, axis=1) == 0).astype(int)
        else:
            head_cam_hist = np.zeros((0, 10), dtype=int)
            tail_cam_hist = np.zeros((0, 10), dtype=int)

        # 1. PRE-CALCULATE HISTORICAL DELAYS FOR EACH NUMBER - Vectorized
        all_delays = []
        for num in range(100):
            appeared_days = np.where(S_history[:, num] == 1)[0]
            if len(appeared_days) == 0:
                gaps = [N + 1]
            else:
                gaps = [int(appeared_days[0] + 1)] + np.diff(appeared_days).astype(int).tolist() + [int(N - appeared_days[-1])]
            all_delays.append(gaps)

        # 2. TRANSITION MATRIX FOR CO-OCCURRENCE (BẠC NHỚ)
        # M[a, b] = P(b today | a yesterday)
        # Using Laplace smoothing
        if N > 1:
            co_occur = S_history[:-1].T @ S_history[1:]
            freq_yesterday = S_history[:-1].sum(axis=0)
            M = (co_occur + 1) / (freq_yesterday[:, None] + 2)
        else:
            M = np.ones((100, 100)) * 0.27

        # 3. CONDITIONAL PROB OF HEAD/TAIL CÂM - Vectorized co-occurrence
        if N > 1:
            head_cam_co_occur = head_cam_hist[:-1].T @ S_history[1:]  # (10, 100)
            head_cam_count = head_cam_hist[:-1].sum(axis=0)  # (10,)
            C_head = (head_cam_co_occur + 1) / (head_cam_count[:, None] + 2)

            tail_cam_co_occur = tail_cam_hist[:-1].T @ S_history[1:]  # (10, 100)
            tail_cam_count = tail_cam_hist[:-1].sum(axis=0)  # (10,)
            C_tail = (tail_cam_co_occur + 1) / (tail_cam_count[:, None] + 2)
        else:
            C_head = np.ones((10, 100)) * 0.27
            C_tail = np.ones((10, 100)) * 0.27

        # 4. CONDITIONAL PROB OF SPECIAL PRIZE SUM
        # S_sum_prob[sum_val, b] = P(b today | yesterday spec sum = sum_val)
        if N > 1 and has_special:
            max_sum = 46
            sum_co_occur = np.zeros((max_sum, 100))
            sum_count = np.zeros(max_sum)
            for i in range(N - 1):
                s_sum = int(specials_sum[i])
                if 0 <= s_sum < max_sum:
                    sum_count[s_sum] += 1
                    sum_co_occur[s_sum] += S_history[i + 1]
            C_sum = (sum_co_occur + 0.1) / (sum_count[:, None] + 0.2)
        else:
            C_sum = np.ones((46, 100)) * 0.27

        # 5. LOOP OVER EACH NUMBER TO POPULATE FEATURES
        for num in range(100):
            f = features[num]

            # --- Delay Features ---
            gaps = all_delays[num]
            current_delay = gaps[-1]

            f['delay'] = current_delay
            f['delay_sq'] = current_delay ** 2
            
            # Historical gaps stats (excluding the current incomplete delay)
            hist_gaps = gaps[:-1] if len(gaps) > 1 else gaps
            f['delay_mean'] = np.mean(hist_gaps)
            f['delay_std'] = np.std(hist_gaps) if len(hist_gaps) > 1 else 1.0
            f['delay_zscore'] = (current_delay - f['delay_mean']) / (f['delay_std'] + 1e-5)
            
            # Momentum of delays
            last_3_gaps = hist_gaps[-3:] if len(hist_gaps) >= 3 else hist_gaps
            f['delay_momentum'] = current_delay - np.mean(last_3_gaps)
            
            # Volatility of delays
            last_10_gaps = hist_gaps[-10:] if len(hist_gaps) >= 10 else hist_gaps
            f['delay_volatility'] = np.std(last_10_gaps) if len(last_10_gaps) > 1 else 1.0
            
            # EWMA of delays
            ewma = hist_gaps[0]
            for g in hist_gaps[1:]:
                ewma = 0.2 * g + 0.8 * ewma
            f['delay_ewma'] = ewma

            # --- Frequency Features (Multi-Timeframe) ---
            for W in [3, 7, 14, 30, 60, 90, 120, 180, 365]:
                if N >= W:
                    f[f'freq_{W}d'] = S_history[-W:, num].mean()
                else:
                    f[f'freq_{W}d'] = S_history[:, num].mean() if N > 0 else 0.27

            # Frequency Momentum
            f['freq_momentum_short'] = f['freq_7d'] - f['freq_30d']
            f['freq_momentum_long'] = f['freq_30d'] - f['freq_90d']

            # Rolling stats of appearances (over last 30 days)
            window_data = S_history[-30:, num] if N >= 30 else S_history[:, num]
            p = np.mean(window_data) if len(window_data) > 0 else 0.27
            f['freq_mean_30'] = p
            f['freq_std_30'] = np.sqrt(p * (1 - p))
            f['freq_skew_30'] = (1 - 2 * p) / (f['freq_std_30'] + 1e-5)
            f['freq_kurt_30'] = (1 - 6 * p * (1 - p)) / (p * (1 - p) + 1e-5)

            # --- Markov Chain Features ---
            if N > 2:
                # Order 1
                x1 = S_history[:-1, num]
                y1 = S_history[1:, num]
                
                state_prev = S_history[-1, num]
                matches_prev_state = (x1 == state_prev)
                numerator = np.sum(y1[matches_prev_state]) + 1
                denominator = np.sum(matches_prev_state) + 2
                f['markov_order1'] = numerator / denominator

                # Order 2
                x2_1 = S_history[:-2, num]
                x2_2 = S_history[1:-1, num]
                y2 = S_history[2:, num]

                state_prev2 = S_history[-2, num]
                matches_prev2_states = (x2_1 == state_prev2) & (x2_2 == state_prev)
                numerator2 = np.sum(y2[matches_prev2_states]) + 1
                denominator2 = np.sum(matches_prev2_states) + 2
                f['markov_order2'] = numerator2 / denominator2

                # Markov transition entropy & persistence
                p01 = (np.sum((x1 == 0) & (y1 == 1)) + 1) / (np.sum(x1 == 0) + 2)
                p11 = (np.sum((x1 == 1) & (y1 == 1)) + 1) / (np.sum(x1 == 1) + 2)
                p00 = 1 - p01
                p10 = 1 - p11
                h0 = -p00 * np.log2(p00 + 1e-9) - p01 * np.log2(p01 + 1e-9)
                h1 = -p10 * np.log2(p10 + 1e-9) - p11 * np.log2(p11 + 1e-9)
                f['markov_entropy'] = (1 - p) * h0 + p * h1
                f['markov_persistence'] = p11 - p01
            else:
                f['markov_order1'] = 0.27
                f['markov_order2'] = 0.27
                f['markov_entropy'] = 1.0
                f['markov_persistence'] = 0.0

            # --- Bayesian / Co-occurrence Features ---
            if N > 0:
                # Average probability given yesterday's active numbers
                yesterday_active = S_history[-1]
                f['cond_prob_yesterday'] = (yesterday_active @ M[:, num]) / (yesterday_active.sum() + 1e-5)

                # Head and tail câm probability
                yesterday_heads_cam = head_cam_hist[-1]
                yesterday_tails_cam = tail_cam_hist[-1]

                active_heads = np.where(yesterday_heads_cam == 1)[0]
                if len(active_heads) > 0:
                    f['cond_prob_head_cam'] = np.mean(C_head[active_heads, num])
                else:
                    f['cond_prob_head_cam'] = p

                active_tails = np.where(yesterday_tails_cam == 1)[0]
                if len(active_tails) > 0:
                    f['cond_prob_tail_cam'] = np.mean(C_tail[active_tails, num])
                else:
                    f['cond_prob_tail_cam'] = p

                # Special prize digit sum probability
                if has_special:
                    yesterday_sum = int(specials_sum[-1])
                    if 0 <= yesterday_sum < len(C_sum):
                        f['cond_prob_spec_sum'] = C_sum[yesterday_sum, num]
                    else:
                        f['cond_prob_spec_sum'] = p
                else:
                    f['cond_prob_spec_sum'] = p
            else:
                f['cond_prob_yesterday'] = 0.27
                f['cond_prob_head_cam'] = 0.27
                f['cond_prob_tail_cam'] = 0.27
                f['cond_prob_spec_sum'] = 0.27

            # --- Pairs, Twins, and Mirror Features ---
            f['is_twin'] = 1 if num // 10 == num % 10 else 0
            
            # Inverted Pair
            inv_num = self.get_inverted(num)
            f['inverted_appeared_yesterday'] = 1 if N > 0 and S_history[-1, inv_num] == 1 else 0
            if N > 1:
                # P(num today | inv_num yesterday)
                inv_yesterday = S_history[:-1, inv_num]
                f['cond_prob_inverted'] = (np.sum((inv_yesterday == 1) & (S_history[1:, num] == 1)) + 1) / (np.sum(inv_yesterday == 1) + 2)
            else:
                f['cond_prob_inverted'] = 0.27

            # Mirror Number
            mir_num = self.get_mirror(num)
            f['mirror_appeared_yesterday'] = 1 if N > 0 and S_history[-1, mir_num] == 1 else 0
            if N > 1:
                # P(num today | mir_num yesterday)
                mir_yesterday = S_history[:-1, mir_num]
                f['cond_prob_mirror'] = (np.sum((mir_yesterday == 1) & (S_history[1:, num] == 1)) + 1) / (np.sum(mir_yesterday == 1) + 2)
            else:
                f['cond_prob_mirror'] = 0.27

            # Gap to Inverted / Mirror
            f['dist_to_inverted'] = abs(num - inv_num)
            f['dist_to_mirror'] = abs(num - mir_num)

            # --- Time Features ---
            f['weekday'] = target_weekday
            f['day_of_month'] = target_day
            f['month'] = target_month
            f['is_weekend'] = 1 if target_weekday in (5, 6) else 0

        # Create DataFrame from features
        df_feat = pd.DataFrame.from_dict(features, orient='index')

        # Add global group rank and percentile features for delay
        delays = df_feat['delay'].values
        df_feat['delay_rank'] = np.argsort(np.argsort(delays))
        df_feat['delay_percentile'] = df_feat['delay_rank'] / 100.0

        return df_feat

    def build_dataset_range(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build full training/testing dataset from start_idx to end_idx.
        Returns X (DataFrame with all features) and y (Series of binary targets).
        """
        prize_cols = [c for c in df.columns if c != 'date']
        total_days = len(df)
        
        # Precompute binary S matrix
        arr_full = df[prize_cols].values.astype(int)
        S_full = np.zeros((total_days, 100), dtype=np.int8)
        rows_f = np.repeat(np.arange(total_days), arr_full.shape[1])
        cols_f = arr_full.flatten()
        valid = (cols_f >= 0) & (cols_f < 100)
        S_full[rows_f[valid], cols_f[valid]] = 1

        all_X = []
        all_y = []

        print(f"Generating features for range [{start_idx} to {end_idx}] ({end_idx - start_idx} days)...")
        for step, idx in enumerate(range(start_idx, end_idx)):
            target_row = df.iloc[idx]
            target_date = pd.to_datetime(target_row['date'])
            
            # History slices
            df_hist = df.iloc[:idx]
            S_hist = S_full[:idx]
            
            # Build features for 100 numbers
            df_feat = self.build_features_for_day(df_hist, S_hist, target_date)
            df_feat['target_num'] = df_feat.index
            df_feat['date'] = str(target_date.date())
            
            # Targets
            y_today = S_full[idx]
            
            all_X.append(df_feat)
            all_y.extend(y_today.tolist())

            if (step + 1) % 50 == 0:
                print(f"  Processed: {step + 1}/{end_idx - start_idx} days...")

        X = pd.concat(all_X, ignore_index=True)
        y = pd.Series(all_y, name='appeared')
        
        return X, y
