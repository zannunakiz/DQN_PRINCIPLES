import random
import math

# Konstanta
STATE_SIZE = 9
ACTION_SIZE = 3
BATCH_SIZE = 64
GAMMA = 0.99

class DQN:
    """Simple DQN model untuk kalkulasi training"""
    def __init__(self):
        # Inisialisasi weights dan biases secara random
        # Struktur: 9 -> 64 -> 64 -> 3
        self.W1 = [[random.uniform(-0.1, 0.1) for _ in range(STATE_SIZE)] for _ in range(64)]
        self.b1 = [random.uniform(-0.1, 0.1) for _ in range(64)]
        self.W2 = [[random.uniform(-0.1, 0.1) for _ in range(64)] for _ in range(64)]
        self.b2 = [random.uniform(-0.1, 0.1) for _ in range(64)]
        self.W3 = [[random.uniform(-0.1, 0.1) for _ in range(64)] for _ in range(3)]
        self.b3 = [random.uniform(-0.1, 0.1) for _ in range(3)]
    
    def relu(self, x):
        return max(0, x)
    
    def forward(self, state):
        # Layer 1: 9 -> 64
        h1 = []
        for i in range(64):
            total = self.b1[i]
            for j in range(STATE_SIZE):
                total += self.W1[i][j] * state[j]
            h1.append(self.relu(total))
        
        # Layer 2: 64 -> 64
        h2 = []
        for i in range(64):
            total = self.b2[i]
            for j in range(64):
                total += self.W2[i][j] * h1[j]
            h2.append(self.relu(total))
        
        # Layer 3: 64 -> 3 (output Q-values)
        q_values = []
        for i in range(3):
            total = self.b3[i]
            for j in range(64):
                total += self.W3[i][j] * h2[j]
            q_values.append(total)
        
        return q_values

def generate_replay_buffer():
    """Generate 64 sample replay buffer sesuai format"""
    replay_buffer = []
    
    # Sample pertama menggunakan data yang diminta
    state1 = [1.0, 0.04, 0.41, 0.91, 0.35, 0.01, 0.73, 0.2, 1.0]
    action1 = 0
    reward1 = -0.98
    next_state1 = [2.0, 0.29, 0.53, 0.09, 0.47, 0.08, 0.5, 0.9, 0.0]
    done1 = 0
    replay_buffer.append([state1, action1, reward1, next_state1, done1])
    
    # Generate 63 sample lainnya dengan variasi
    for i in range(BATCH_SIZE - 1):
        # Random state
        angle = random.randint(1, 5)
        sensors = [round(random.uniform(0.0, 1.0), 2) for _ in range(7)]
        prev_action = random.randint(0, 3)
        state = [float(angle)] + sensors + [float(prev_action)]
        
        # Random action
        action = random.randint(0, 2)
        
        # Random reward
        reward = round(random.uniform(-2.0, 1.5), 2)
        
        # Random next state
        next_angle = random.randint(1, 5)
        next_sensors = [round(random.uniform(0.0, 1.0), 2) for _ in range(7)]
        next_prev_action = action
        next_state = [float(next_angle)] + next_sensors + [float(next_prev_action)]
        
        # Random done
        done = 1 if random.random() < 0.1 else 0
        
        replay_buffer.append([state, action, reward, next_state, done])
    
    return replay_buffer

def predict_q_values(model, state):
    """Prediksi Q-values dari state menggunakan model"""
    return model.forward(state)

def calculate_target(reward, done, max_q_next):
    """Hitung target Q-value"""
    return reward + (1 - done) * GAMMA * max_q_next

def main():
    print("=" * 80)
    print("CALC_Q-TRAINING.PY - DQN Training Calculator")
    print("=" * 80)
    
    # A) Check Replay Buffer Size
    print("\nA) Check Replay Buffer Size D(size)")
    print("-" * 60)
    replay_buffer = generate_replay_buffer()
    print(f"Replay Buffer Size: {len(replay_buffer)}")
    print("D(size) >= 64: Start Training")
    print("D(size) < 64: No Training, continue gathering information")
    print(f"Dikarenakan D(size) = {len(replay_buffer)} maka Start Training.")
    print("-" * 60)
    
    # B) Pengambilan Mini-Batch
    print("\nB) Pengambilan Mini-Batch")
    print("-" * 60)
    print("Replay buffer menyimpan data transisi dalam bentuk")
    print("D = (s_t, a_t, r_t, s_(t+1), d_t)")
    print(f"Batch Size = {BATCH_SIZE}")
    print("64 sampel dipilih secara acak (mini-batch) dari replay buffer.")
    
    # Ambil 64 sampel acak
    mini_batch = random.sample(replay_buffer, BATCH_SIZE)
    
    # Tampilkan contoh satu data (sample pertama yang sudah ditentukan)
    sample = mini_batch[0]
    print("\nContoh salah satu data pada mini-batch:")
    print(f"s_t      : {sample[0]} (state step 1)")
    print(f"a_(t)    : {sample[1]} (action step 1)")
    print(f"r_(t)    : {sample[2]} (reward step 1)")
    print(f"s_(t+1)  : {sample[3]} (state step 2)")
    print(f"d_t      : {sample[4]} (episode {'berakhir' if sample[4] == 1 else 'belum berakhir'})")
    print("-" * 60)
    
    # Initialize model
    print("\nInisialisasi Model")
    print("-" * 60)
    model = DQN()
    print("Model DQN diinisialisasi dengan struktur 9 -> 64 -> 64 -> 3")
    print("Weights dan biases diinisialisasi secara random.")
    print("-" * 60)
    
    # Target Q-Value, TD Error, Loss Function
    print("\nTarget Q-Value, TD Error, Loss Function")
    print("-" * 60)
    print("Target Q-value adalah nilai target yang dijadikan acuan untuk update Q-network.")
    print("Rumus: y_i = r_t + (1-d_t) * gamma * max Q(s_(t+1))")
    print(f"Gamma = {GAMMA}")
    print("-" * 60)
    
    # ==================== HITUNG MANUAL UNTUK SAMPLE PERTAMA ====================
    print("\n" + "=" * 80)
    print("PERHITUNGAN MANUAL UNTUK SAMPLE 1")
    print("=" * 80)
    
    # Ambil sample pertama
    sample_manual = mini_batch[0]
    state_manual = sample_manual[0]
    action_manual = sample_manual[1]
    reward_manual = sample_manual[2]
    next_state_manual = sample_manual[3]
    done_manual = sample_manual[4]
    
    print("\nData Sample 1:")
    print(f"s_t      : {state_manual}")
    print(f"a_(t)    : {action_manual} (action step 1)")
    print(f"r_(t)    : {reward_manual} (reward step 1)")
    print(f"s_(t+1)  : {next_state_manual} (state step 2)")
    print(f"d_t      : {done_manual} (episode {'berakhir' if done_manual == 1 else 'belum berakhir'})")
    
    # Prediksi Q-values untuk state saat ini
    q_values_manual = predict_q_values(model, state_manual)
    q0_manual, q1_manual, q2_manual = q_values_manual[0], q_values_manual[1], q_values_manual[2]
    q_predicted_manual = q_values_manual[action_manual]
    
    # Prediksi Q-values untuk next state
    q_next_values_manual = predict_q_values(model, next_state_manual)
    max_q_next_manual = max(q_next_values_manual)
    
    print("\nA) Predicted Q-values untuk s_t:")
    print(f"   Q(s_t) = [{q0_manual:.4f}, {q1_manual:.4f}, {q2_manual:.4f}]")
    print(f"   Q_predicted (untuk aksi {action_manual}) = {q_predicted_manual:.4f}")
    
    print("\nB) Predicted Q-values untuk s_(t+1):")
    print(f"   Q(s_(t+1)) = [{q_next_values_manual[0]:.4f}, {q_next_values_manual[1]:.4f}, {q_next_values_manual[2]:.4f}]")
    print(f"   max Q(s_(t+1)) = {max_q_next_manual:.4f}")
    
    # Hitung Target Q-value
    print("\nC) Target Q-value (y_i):")
    print(f"   Diketahui:")
    print(f"   r_t = {reward_manual}")
    print(f"   d_t = {done_manual}")
    print(f"   γ = {GAMMA}")
    print(f"   max Q(s_(t+1)) = {max_q_next_manual:.4f}")
    print(f"   Rumus: y_i = r_t + (1-d_t) * γ * max Q(s_(t+1))")
    print(f"   Substitusi:")
    print(f"   y_i = {reward_manual} + (1-{done_manual}) * {GAMMA} * {max_q_next_manual:.4f}")
    
    target_manual = calculate_target(reward_manual, done_manual, max_q_next_manual)
    print(f"   y_i = {reward_manual} + {1-done_manual} * {GAMMA * max_q_next_manual:.4f}")
    print(f"   y_i = {reward_manual} + {((1-done_manual) * GAMMA * max_q_next_manual):.4f}")
    print(f"   y_i = {target_manual:.4f}")
    
    # Hitung TD Error
    print("\nD) TD Error (δ):")
    print(f"   Diketahui:")
    print(f"   y_i = {target_manual:.4f}")
    print(f"   Q_predicted = {q_predicted_manual:.4f}")
    print(f"   Rumus: δ = y_i - Q_predicted")
    print(f"   δ = {target_manual:.4f} - {q_predicted_manual:.4f}")
    
    td_error_manual = target_manual - q_predicted_manual
    print(f"   δ = {td_error_manual:.4f}")
    
    # Hitung Error²
    print("\nE) Error²:")
    print(f"   Rumus: Error² = (δ)²")
    print(f"   Error² = ({td_error_manual:.4f})²")
    
    error_squared_manual = td_error_manual ** 2
    print(f"   Error² = {error_squared_manual:.4f}")
    
    print("\n" + "=" * 80)
    print("RINGKASAN HASIL PERHITUNGAN SAMPLE 1:")
    print("-" * 60)
    print(f"Target Q-value (y_i)    : {target_manual:.4f}")
    print(f"Q_predicted             : {q_predicted_manual:.4f}")
    print(f"TD Error (δ)            : {td_error_manual:.4f}")
    print(f"Error²                  : {error_squared_manual:.4f}")
    print("=" * 80)
    
    # ==================== TABEL 64 SAMPLES ====================
    print("\n\nTabel Perhitungan 64 Mini-Batch:")
    print("-" * 170)
    print(f"{'Sampel':<6} {'Q0':<10} {'Q1':<10} {'Q2':<10} {'Action':<6} {'Q_pred':<10} {'Target (y_i)':<15} {'TD Error':<12} {'Error²':<12}")
    print("-" * 170)
    
    total_error_squared = 0.0
    results = []
    
    for idx, sample in enumerate(mini_batch, 1):
        state = sample[0]
        action = sample[1]
        reward = sample[2]
        next_state = sample[3]
        done = sample[4]
        
        # Prediksi Q-values untuk state saat ini
        q_values = predict_q_values(model, state)
        q0, q1, q2 = q_values[0], q_values[1], q_values[2]
        q_predicted = q_values[action]
        
        # Prediksi Q-values untuk next state
        q_next_values = predict_q_values(model, next_state)
        max_q_next = max(q_next_values)
        
        # Hitung Target Q-value
        target = calculate_target(reward, done, max_q_next)
        
        # Hitung TD Error
        td_error = target - q_predicted
        
        # Hitung Error²
        error_squared = td_error ** 2
        
        total_error_squared += error_squared
        
        results.append({
            'sample': idx,
            'q0': q0,
            'q1': q1,
            'q2': q2,
            'action': action,
            'q_predicted': q_predicted,
            'target': target,
            'td_error': td_error,
            'error_squared': error_squared
        })
        
        # Print tabel dengan format rapi
        action_names = ["F", "L", "R"]
        print(f"{idx:<6} {q0:<10.4f} {q1:<10.4f} {q2:<10.4f} {action_names[action]:<6} {q_predicted:<10.4f} {target:<15.4f} {td_error:<12.4f} {error_squared:<12.4f}")
    
    print("-" * 170)
    
    # Hitung Loss (MSE)
    loss = total_error_squared / BATCH_SIZE
    
    print("\nPerhitungan Loss Function (MSE):")
    print("-" * 60)
    print(f"Σ Error² = {total_error_squared:.4f}")
    print(f"B = {BATCH_SIZE}")
    print(f"L = Σ Error² / B")
    print(f"L = {total_error_squared:.4f} / {BATCH_SIZE}")
    print(f"L = {loss:.6f}")
    print("-" * 60)
    
    # Interpretasi Loss
    print("\nInterpretasi Loss:")
    print("-" * 60)
    if loss < 0.5:
        print("✓ Loss kecil → Q-Network semakin dekat ke Target Network")
        print("  Prediksi Q-value cukup akurat")
        print("  Update gradien akan kecil")
    elif loss < 1.0:
        print("⚠ Loss sedang → Q-Network cukup dekat ke Target Network")
        print("  Prediksi Q-value cukup akurat")
        print("  Update gradien moderat")
    else:
        print("✗ Loss besar → Q-Network masih jauh berbeda dari Target Network")
        print("  Diperlukan backpropagation untuk memperbaiki weights")
        print("  Update gradien akan lebih besar")
    print("-" * 60)
    
    # Statistik tambahan
    print("\nStatistik Tambahan:")
    print("-" * 60)
    targets = [r['target'] for r in results]
    q_preds = [r['q_predicted'] for r in results]
    td_errors = [r['td_error'] for r in results]
    
    print(f"Target Q-Value   - Min: {min(targets):.4f}, Max: {max(targets):.4f}, Mean: {sum(targets)/len(targets):.4f}")
    print(f"Q Predicted      - Min: {min(q_preds):.4f}, Max: {max(q_preds):.4f}, Mean: {sum(q_preds)/len(q_preds):.4f}")
    print(f"TD Error         - Min: {min(td_errors):.4f}, Max: {max(td_errors):.4f}, Mean: {sum(td_errors)/len(td_errors):.4f}")
    print(f"Error²           - Min: {min([r['error_squared'] for r in results]):.4f}, Max: {max([r['error_squared'] for r in results]):.4f}")
    print("-" * 60)
    
    print("\n" + "=" * 80)
    print(f"HASIL LOSS VALUE (MSE): {loss:.6f}")
    print("=" * 80)
    
    print("\nKesimpulan:")
    print("Loss function menggunakan Mean Squared Error (MSE) antara Target Q-value")
    print("dan Predicted Q-value pada seluruh sampel minibatch. Nilai loss ini")
    print("kemudian digunakan dalam Backpropagation untuk memperbarui Q-Network.")
    print("=" * 80)

if __name__ == "__main__":
    main()