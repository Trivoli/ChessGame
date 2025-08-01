import chess
import random
from collections import Counter, defaultdict
import pygame
import sys
import os
import json
import datetime
import threading
import time
import math

PLAYER_MODEL_FILE = 'player_model.json'
PLAYER_GAMES_LOG = 'player_games.log'

# --- Enhanced Visual Settings ---
BOARD_THEMES = {
    'Classic': [(240, 217, 181), (181, 136, 99)],
    'Blue': [(240, 248, 255), (100, 149, 237)],
    'Green': [(240, 255, 240), (34, 139, 34)],
    'Purple': [(245, 240, 255), (138, 43, 226)],
    'Wood': [(222, 184, 135), (139, 69, 19)],
    'Ocean': [(176, 224, 230), (25, 25, 112)],
    'Sunset': [(255, 228, 196), (255, 140, 0)]
}

HIGHLIGHT_COLORS = {
    'Classic': (100, 200, 100),
    'Bright': (255, 255, 0),
    'Soft': (144, 238, 144),
    'Red': (255, 99, 71),
    'Blue': (135, 206, 250)
}

# Game settings with defaults
GAME_SETTINGS = {
    'board_theme': 'Classic',
    'highlight_color': 'Classic',
    'show_legal_moves': True,
    'animate_moves': True,
    'show_coordinates': True,
    'show_captured_pieces': True,
    'show_move_history': True,
    'board_flipped': False,
    'sound_enabled': False,
    'auto_save': True,
    'move_delay': 0.2  # Reduced for faster gameplay
}

# --- Persistent Player Model ---
class PlayerModel:
    def __init__(self):
        self.move_counter = Counter()
        self.opening_sequences = defaultdict(Counter)
        self.games_played = 0
        self.cleanup_old_format()
        self.load()
        self.load_from_logs()
        print(f"Loaded {self.games_played} games from history")

    def record_game(self, moves, completed=True):
        for i, move in enumerate(moves):
            self.move_counter[move] += 1
            self.opening_sequences[i][move] += 1
        self.games_played += 1
        self.log_game(moves, completed)

    def log_game(self, moves, completed=True):
        timestamp = datetime.datetime.now().isoformat()
        moves_str = ','.join(moves)
        status = "completed" if completed else "incomplete"
        with open(PLAYER_GAMES_LOG, 'a') as f:
            f.write(f"[{timestamp}] {moves_str} ({status})\n")

    def save(self, moves=None, completed=True):
        if moves is not None:
            timestamp = datetime.datetime.now().isoformat()
            game_obj = {
                'timestamp': timestamp,
                'moves': moves,
                'completed': completed
            }
            with open(PLAYER_MODEL_FILE, 'a') as f:
                f.write(json.dumps(game_obj) + '\n')

    def load(self):
        if os.path.exists(PLAYER_MODEL_FILE):
            with open(PLAYER_MODEL_FILE, 'r') as f:
                content = f.read().strip()
                if content.startswith('{') and content.endswith('}'):
                    try:
                        data = json.loads(content)
                        self.move_counter = Counter(data.get('move_counter', {}))
                        self.opening_sequences = defaultdict(Counter, {int(k): Counter(v) for k, v in data.get('opening_sequences', {}).items()})
                        self.games_played = data.get('games_played', 0)
                    except Exception:
                        pass
                else:
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                game = json.loads(line)
                                moves = game.get('moves', [])
                                for i, move in enumerate(moves):
                                    self.move_counter[move] += 1
                                    self.opening_sequences[i][move] += 1
                                if moves:
                                    self.games_played += 1
                            except Exception:
                                continue

    def load_from_logs(self):
        if os.path.exists(PLAYER_GAMES_LOG):
            with open(PLAYER_GAMES_LOG, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and line.startswith('['):
                        try:
                            moves_start = line.find(']') + 1
                            moves_end = line.find(' (')
                            if moves_start > 0 and moves_end > moves_start:
                                moves_str = line[moves_start:moves_end].strip()
                                moves = moves_str.split(',') if moves_str else []

                                for i, move in enumerate(moves):
                                    if move:
                                        self.move_counter[move] += 1
                                        self.opening_sequences[i][move] += 1
                                if moves:
                                    self.games_played += 1
                        except Exception as e:
                            print(f"Error parsing log line: {e}")
                            continue

    def most_common_opening(self, move_number):
        if move_number in self.opening_sequences:
            return self.opening_sequences[move_number].most_common(1)[0][0]
        return None

    def cleanup_old_format(self):
        if os.path.exists(PLAYER_MODEL_FILE):
            try:
                with open(PLAYER_MODEL_FILE, 'r') as f:
                    content = f.read().strip()
                    if content.startswith('{') and content.endswith('}'):
                        data = json.loads(content)
                        backup_file = PLAYER_MODEL_FILE + '.backup'
                        with open(backup_file, 'w') as bf:
                            bf.write(content)

                        with open(PLAYER_MODEL_FILE, 'w') as f:
                            f.write('')

                        self.move_counter = Counter(data.get('move_counter', {}))
                        self.opening_sequences = defaultdict(Counter, {int(k): Counter(v) for k, v in data.get('opening_sequences', {}).items()})
                        self.games_played = data.get('games_played', 0)
                        print(f"Converted old format to new format. Backup saved as {backup_file}")
            except Exception as e:
                print(f"Error during format conversion: {e}")

# --- Enhanced Chess Engine with Advanced Optimizations ---

# Piece values with enhanced endgame considerations
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# Piece-Square Tables for positional evaluation
# Pawn table (White perspective, black will be flipped)
PAWN_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
   50, 50, 50, 50, 50, 50, 50, 50,
   10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

# Knight table
KNIGHT_TABLE = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
]

# Bishop table
BISHOP_TABLE = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
]

# Rook table
ROOK_TABLE = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

# Queen table
QUEEN_TABLE = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
]

# King middle game table
KING_MG_TABLE = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

# King endgame table
KING_EG_TABLE = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50
]

PIECE_SQUARE_TABLES = {
    chess.PAWN: PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK: ROOK_TABLE,
    chess.QUEEN: QUEEN_TABLE,
    chess.KING: KING_MG_TABLE  # Will switch to EG in endgame
}

# Transposition Table Entry
class TTEntry:
    def __init__(self, depth, score, flag, best_move=None):
        self.depth = depth
        self.score = score
        self.flag = flag  # EXACT, LOWER_BOUND, UPPER_BOUND
        self.best_move = best_move

# Transposition Table Flags
TT_EXACT = 0
TT_LOWER_BOUND = 1
TT_UPPER_BOUND = 2

# Enhanced Chess AI with all optimizations
class OptimizedChessEngine:
    def __init__(self):
        self.transposition_table = {}
        self.killer_moves = [[] for _ in range(20)]  # Killer moves for each depth
        self.history_table = defaultdict(int)  # History heuristic
        self.nodes_searched = 0
        self.tt_hits = 0
        
    def clear_cache(self):
        """Clear search caches"""
        self.transposition_table.clear()
        self.killer_moves = [[] for _ in range(20)]
        self.history_table.clear()
        
    def get_piece_square_value(self, piece, square, is_endgame=False):
        """Get piece-square table value"""
        piece_type = piece.piece_type
        color = piece.color
        
        if piece_type == chess.KING and is_endgame:
            table = KING_EG_TABLE
        else:
            table = PIECE_SQUARE_TABLES.get(piece_type, [0] * 64)
        
        # For black pieces, flip the square
        if color == chess.BLACK:
            square = square ^ 56  # Flip rank
            
        return table[square]
    
    def is_endgame(self, board):
        """Enhanced endgame detection"""
        piece_count = len(board.piece_map())
        if piece_count <= 10:
            return True
            
        # Check if queens are off the board
        queens = len([p for p in board.piece_map().values() if p.piece_type == chess.QUEEN])
        if queens == 0:
            return True
            
        # Check for light piece endgames
        major_pieces = len([p for p in board.piece_map().values() 
                          if p.piece_type in [chess.QUEEN, chess.ROOK]])
        if major_pieces <= 2:
            return True
            
        return False
    
    def evaluate_board(self, board, aggressivity_factor=1.0):
        """Enhanced evaluation function with piece-square tables and positional factors"""
        if board.is_checkmate():
            return -30000 if board.turn else 30000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        is_endgame = self.is_endgame(board)
        material_score = 0
        positional_score = 0
        
        # Material and positional evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Material value
                piece_value = PIECE_VALUES[piece.piece_type]
                
                # Piece-square bonus
                positional_bonus = self.get_piece_square_value(piece, square, is_endgame)
                
                if piece.color == chess.WHITE:
                    material_score += piece_value
                    positional_score += positional_bonus
                else:
                    material_score -= piece_value
                    positional_score -= positional_bonus
        
        # Mobility and tactical bonuses
        mobility_score = 0
        current_turn = board.turn
        
        # White mobility
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        
        # Black mobility  
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        
        board.turn = current_turn
        mobility_score = (white_mobility - black_mobility) * 10
        
        # Center control bonus
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        extended_center = [chess.C3, chess.C4, chess.C5, chess.C6,
                          chess.D3, chess.D6, chess.E3, chess.E6,
                          chess.F3, chess.F4, chess.F5, chess.F6]
        
        center_control = 0
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                value = 20 if piece.color == chess.WHITE else -20
                center_control += value
                
        for square in extended_center:
            piece = board.piece_at(square)
            if piece:
                value = 10 if piece.color == chess.WHITE else -10
                center_control += value
        
        # King safety (more important in middlegame)
        king_safety = 0
        if not is_endgame:
            white_king = board.king(chess.WHITE)
            black_king = board.king(chess.BLACK)
            
            if white_king:
                white_king_attackers = len(board.attackers(chess.BLACK, white_king))
                king_safety -= white_king_attackers * 30
                
            if black_king:
                black_king_attackers = len(board.attackers(chess.WHITE, black_king))
                king_safety += black_king_attackers * 30
        
        # Pawn structure bonuses
        pawn_structure = self.evaluate_pawn_structure(board)
        
        total_score = (material_score + 
                      positional_score + 
                      mobility_score + 
                      center_control + 
                      king_safety + 
                      pawn_structure)
        
        # Apply aggressivity factor
        if aggressivity_factor != 1.0:
            tactical_bonus = (mobility_score + king_safety) * (aggressivity_factor - 1.0)
            total_score += tactical_bonus
            
        return total_score
    
    def evaluate_pawn_structure(self, board):
        """Evaluate pawn structure"""
        score = 0
        white_pawns = []
        black_pawns = []
        
        # Collect pawn positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE:
                    white_pawns.append(square)
                else:
                    black_pawns.append(square)
        
        # Doubled pawns penalty
        white_files = [chess.square_file(sq) for sq in white_pawns]
        black_files = [chess.square_file(sq) for sq in black_pawns]
        
        for file in range(8):
            white_count = white_files.count(file)
            black_count = black_files.count(file)
            if white_count > 1:
                score -= (white_count - 1) * 20
            if black_count > 1:
                score += (black_count - 1) * 20
        
        # Isolated pawns penalty
        for pawn_sq in white_pawns:
            file = chess.square_file(pawn_sq)
            adjacent_files = [file - 1, file + 1]
            has_support = any(chess.square_file(sq) in adjacent_files for sq in white_pawns)
            if not has_support:
                score -= 25
                
        for pawn_sq in black_pawns:
            file = chess.square_file(pawn_sq)
            adjacent_files = [file - 1, file + 1]
            has_support = any(chess.square_file(sq) in adjacent_files for sq in black_pawns)
            if not has_support:
                score += 25
        
        # Passed pawns bonus
        for pawn_sq in white_pawns:
            if self.is_passed_pawn(board, pawn_sq, chess.WHITE):
                rank = chess.square_rank(pawn_sq)
                score += (rank - 1) * 20  # More valuable closer to promotion
                
        for pawn_sq in black_pawns:
            if self.is_passed_pawn(board, pawn_sq, chess.BLACK):
                rank = chess.square_rank(pawn_sq)
                score -= (6 - rank) * 20  # More valuable closer to promotion
        
        return score
    
    def is_passed_pawn(self, board, pawn_square, color):
        """Check if pawn is passed"""
        file = chess.square_file(pawn_square)
        rank = chess.square_rank(pawn_square)
        
        # Check files in front of pawn
        check_files = [file - 1, file, file + 1]
        check_files = [f for f in check_files if 0 <= f <= 7]
        
        if color == chess.WHITE:
            check_ranks = range(rank + 1, 8)
        else:
            check_ranks = range(0, rank)
            
        for check_rank in check_ranks:
            for check_file in check_files:
                square = chess.square(check_file, check_rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color != color:
                    return False
        return True
    
    def order_moves(self, board, moves, best_move=None, depth=0):
        """Enhanced move ordering for better alpha-beta pruning"""
        move_scores = []
        
        for move in moves:
            score = 0
            
            # Best move from transposition table gets highest priority
            if best_move and move == best_move:
                score += 10000
                
            # Captures - use MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                attacking_piece = board.piece_at(move.from_square)
                if captured_piece and attacking_piece:
                    score += PIECE_VALUES[captured_piece.piece_type] * 10
                    score -= PIECE_VALUES[attacking_piece.piece_type]
            
            # Promotions
            if move.promotion:
                score += PIECE_VALUES[move.promotion] * 8
                
            # Checks
            board.push(move)
            if board.is_check():
                score += 500
            board.pop()
            
            # Killer moves
            if depth < len(self.killer_moves):
                if move in self.killer_moves[depth]:
                    score += 100
                    
            # History heuristic
            score += self.history_table[move.uci()]
            
            # Castling bonus
            if board.is_castling(move):
                score += 300
                
            move_scores.append((move, score))
        
        # Sort by score (descending)
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return [move for move, score in move_scores]
    
    def store_killer_move(self, move, depth):
        """Store killer move"""
        if depth < len(self.killer_moves):
            if move not in self.killer_moves[depth]:
                self.killer_moves[depth].append(move)
                if len(self.killer_moves[depth]) > 2:  # Keep only 2 killer moves per depth
                    self.killer_moves[depth].pop(0)
    
    def minimax_optimized(self, board, depth, alpha, beta, maximizing, 
                         player_model=None, move_number=0, aggressivity_factor=1.0):
        """Optimized minimax with all enhancements"""
        self.nodes_searched += 1
        original_alpha = alpha
        
        # Transposition table lookup
        board_hash = hash(str(board))
        if board_hash in self.transposition_table:
            tt_entry = self.transposition_table[board_hash]
            if tt_entry.depth >= depth:
                self.tt_hits += 1
                if tt_entry.flag == TT_EXACT:
                    return tt_entry.score, tt_entry.best_move
                elif tt_entry.flag == TT_LOWER_BOUND:
                    alpha = max(alpha, tt_entry.score)
                elif tt_entry.flag == TT_UPPER_BOUND:
                    beta = min(beta, tt_entry.score)
                    
                if alpha >= beta:
                    return tt_entry.score, tt_entry.best_move
        
        # Terminal node evaluation
        if depth == 0 or board.is_game_over():
            score = self.evaluate_board(board, aggressivity_factor)
            return score, None
        
        best_move = None
        best_score = -float('inf') if maximizing else float('inf')
        
        # Get best move from transposition table for move ordering
        tt_best_move = None
        if board_hash in self.transposition_table:
            tt_best_move = self.transposition_table[board_hash].best_move
        
        # Move ordering
        legal_moves = list(board.legal_moves)
        ordered_moves = self.order_moves(board, legal_moves, tt_best_move, depth)
        
        moves_searched = 0
        for move in ordered_moves:
            board.push(move)
            
            # Player pattern counter-strategy
            move_score_adjustment = 0
            if player_model and not maximizing:  # AI is typically not maximizing (playing as black)
                common_opening = player_model.most_common_opening(move_number)
                if common_opening and move.uci() == common_opening:
                    move_score_adjustment = -50  # Penalty for predictable moves
            
            if maximizing:
                score, _ = self.minimax_optimized(board, depth - 1, alpha, beta, False, 
                                                player_model, move_number + 1, aggressivity_factor)
                score += move_score_adjustment
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
                alpha = max(alpha, score)
            else:
                score, _ = self.minimax_optimized(board, depth - 1, alpha, beta, True, 
                                                player_model, move_number + 1, aggressivity_factor)
                score += move_score_adjustment
                
                if score < best_score:
                    best_score = score
                    best_move = move
                    
                beta = min(beta, score)
            
            board.pop()
            moves_searched += 1
            
            # Alpha-beta pruning
            if alpha >= beta:
                # Store killer move
                if not board.is_capture(move):
                    self.store_killer_move(move, depth)
                # Update history table
                self.history_table[move.uci()] += depth * depth
                break
        
        # Store in transposition table
        tt_flag = TT_EXACT
        if best_score <= original_alpha:
            tt_flag = TT_UPPER_BOUND
        elif best_score >= beta:
            tt_flag = TT_LOWER_BOUND
            
        self.transposition_table[board_hash] = TTEntry(depth, best_score, tt_flag, best_move)
        
        # Limit transposition table size
        if len(self.transposition_table) > 100000:
            # Remove random entries (simple cleanup)
            keys_to_remove = list(self.transposition_table.keys())[:10000]
            for key in keys_to_remove:
                del self.transposition_table[key]
        
        return best_score, best_move
    
    def iterative_deepening_search(self, board, max_depth, time_limit=None, 
                                  player_model=None, move_number=0, aggressivity_factor=1.0):
        """Iterative deepening with time management"""
        start_time = time.time()
        best_move = None
        best_score = 0
        
        for depth in range(1, max_depth + 1):
            if time_limit and (time.time() - start_time) > time_limit:
                break
                
            self.nodes_searched = 0
            self.tt_hits = 0
            
            try:
                score, move = self.minimax_optimized(
                    board, depth, -float('inf'), float('inf'), 
                    board.turn == chess.WHITE, player_model, move_number, aggressivity_factor
                )
                
                if move:
                    best_move = move
                    best_score = score
                    
                # Debug info
                elapsed = time.time() - start_time
                print(f"Depth {depth}: Score {score}, Move {move}, "
                      f"Nodes {self.nodes_searched}, TT hits {self.tt_hits}, "
                      f"Time {elapsed:.2f}s")
                      
            except Exception as e:
                print(f"Error at depth {depth}: {e}")
                break
                
            # If we found a mate, no need to search deeper
            if abs(best_score) > 25000:
                break
        
        return best_score, best_move

# Create global engine instance
engine = OptimizedChessEngine()

def evaluate_board(board, aggressivity_factor=1.0):
    """Wrapper function for compatibility"""
    return engine.evaluate_board(board, aggressivity_factor)

def minimax(board, depth, alpha, beta, maximizing, player_model=None, move_number=0, aggressivity_factor=1.0):
    """Wrapper function for compatibility"""
    return engine.minimax_optimized(board, depth, alpha, beta, maximizing, 
                                   player_model, move_number, aggressivity_factor)

# --- Adaptive AI with Enhanced Engine ---

# --- Minimax and Evaluation ---

# --- Enhanced Adaptive AI with Optimized Engine ---
class AdvancedAdaptiveChessAI:
    def __init__(self, player_model, search_depth=3, aggressivity_factor=1.0):
        self.player_model = player_model
        self.current_game_moves = []
        self.search_depth = search_depth
        self.aggressivity_factor = aggressivity_factor
        self.thinking = False
        self.ai_move = None
        self.thinking_thread = None
        self.engine = engine  # Use the global optimized engine
        self.last_search_time = 0
        self.adaptive_depth = search_depth
        self.position_history = []  # Track position repetitions
        
    def get_adaptive_depth(self, board):
        """Dynamically adjust search depth based on game phase and complexity"""
        base_depth = self.search_depth
        
        # Endgame gets deeper search
        if self.engine.is_endgame(board):
            return min(base_depth + 2, 8)
        
        # Opening gets standard depth
        if board.fullmove_number <= 10:
            return base_depth
        
        # Complex middle game positions get deeper search
        legal_moves = len(list(board.legal_moves))
        if legal_moves > 35:  # Complex position
            return max(base_depth - 1, 2)
        elif legal_moves < 20:  # Simplified position
            return min(base_depth + 1, 7)
            
        return base_depth
    
    def get_time_limit(self, board):
        """Get appropriate time limit for the current position"""
        base_time = 2.0  # 2 seconds base time
        
        # Less time in opening
        if board.fullmove_number <= 10:
            return base_time * 0.5
            
        # More time for critical positions
        if board.is_check():
            return base_time * 1.5
            
        # More time in endgame
        if self.engine.is_endgame(board):
            return base_time * 1.2
            
        # More time for complex positions
        legal_moves = len(list(board.legal_moves))
        if legal_moves > 35:
            return base_time * 0.8  # Less time for complex positions to avoid timeout
        elif legal_moves < 15:
            return base_time * 1.3  # More time for critical positions
            
        return base_time

    def record_player_move(self, move):
        self.current_game_moves.append(move)

    def detect_repetition(self, board):
        """Detect if we're heading toward a repetition"""
        position_key = str(board)
        self.position_history.append(position_key)
        
        # Keep only last 20 positions
        if len(self.position_history) > 20:
            self.position_history.pop(0)
            
        # Count occurrences of current position
        count = self.position_history.count(position_key)
        return count >= 2

    def calculate_move(self, board_copy):
        try:
            start_time = time.time()
            
            # Clear engine caches periodically to prevent memory issues
            if len(self.current_game_moves) % 20 == 0:
                self.engine.clear_cache()
            
            # Get adaptive depth and time limit
            max_depth = self.get_adaptive_depth(board_copy)
            time_limit = self.get_time_limit(board_copy)
            
            print(f"AI thinking: Depth {max_depth}, Time limit {time_limit}s")
            
            # Use iterative deepening search
            _, best_move = self.engine.iterative_deepening_search(
                board_copy, 
                max_depth, 
                time_limit,
                self.player_model, 
                board_copy.fullmove_number - 1, 
                self.aggressivity_factor
            )

            # Fallback to simple minimax if iterative deepening fails
            if not best_move:
                print("Fallback to simple minimax")
                _, best_move = self.engine.minimax_optimized(
                    board_copy, 
                    min(max_depth, 4), 
                    -float('inf'), 
                    float('inf'), 
                    board_copy.turn == chess.WHITE, 
                    self.player_model, 
                    board_copy.fullmove_number - 1, 
                    self.aggressivity_factor
                )

            if best_move:
                # Check for repetition avoidance
                if self.detect_repetition(board_copy):
                    # Try to find a different move if we're repeating
                    legal_moves = list(board_copy.legal_moves)
                    if len(legal_moves) > 1:
                        legal_moves.remove(best_move)
                        # Quick evaluation of alternative moves
                        alternative_move = random.choice(legal_moves)
                        print(f"Avoiding repetition, chose alternative: {alternative_move}")
                        best_move = alternative_move
                
                self.ai_move = best_move
                self.last_search_time = time.time() - start_time
                print(f"AI found move: {best_move} in {self.last_search_time:.2f}s")
            else:
                # Final fallback to random move
                legal_moves = list(board_copy.legal_moves)
                if legal_moves:
                    self.ai_move = random.choice(legal_moves)
                    print("AI fallback to random move")
                else:
                    self.ai_move = None
                    
        except Exception as e:
            print(f"AI calculation error: {e}")
            legal_moves = list(board_copy.legal_moves)
            if legal_moves:
                self.ai_move = random.choice(legal_moves)
            else:
                self.ai_move = None
        finally:
            self.thinking = False

    def start_thinking(self, board):
        if not self.thinking:
            self.thinking = True
            self.ai_move = None
            board_copy = board.copy()
            self.thinking_thread = threading.Thread(target=self.calculate_move, args=(board_copy,))
            self.thinking_thread.daemon = True
            self.thinking_thread.start()

    def get_move(self):
        if not self.thinking and self.ai_move is not None:
            move = self.ai_move
            self.ai_move = None
            return move
        return None

    def end_game(self, completed=True):
        self.player_model.record_game(self.current_game_moves, completed)
        self.player_model.save(self.current_game_moves, completed)
        self.current_game_moves = []
        self.position_history = []  # Clear position history
        
    def get_strength_description(self):
        """Get a description of current AI strength"""
        descriptions = {
            1: "Beginner (Fast, Basic)",
            2: "Novice (Quick Tactics)", 
            3: "Intermediate (Balanced)",
            4: "Advanced (Strong Tactics)",
            5: "Expert (Deep Analysis)",
            6: "Master (Tournament Strength)"
        }
        return descriptions.get(self.search_depth, f"Custom Depth {self.search_depth}")
        
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'last_search_time': self.last_search_time,
            'transposition_hits': self.engine.tt_hits,
            'nodes_searched': self.engine.nodes_searched,
            'adaptive_depth': self.adaptive_depth,
            'tt_size': len(self.engine.transposition_table)
        }

# --- Enhanced Pygame Chess GUI ---
WHITE_TURN_BG = (240, 248, 255)
WHITE_TURN_TEXT = (25, 25, 112)
BLACK_TURN_BG = (47, 79, 79)
BLACK_TURN_TEXT = (255, 255, 255)
# Optimized for MacBook M1 Air screen
BOARD_WIDTH = 560  # Reduced from 640
BOARD_HEIGHT = 560  # Reduced from 640
PANEL_HEIGHT = 80   # Reduced from 100
SIDE_PANEL_WIDTH = 220  # Reduced from 250
WIDTH = BOARD_WIDTH + SIDE_PANEL_WIDTH
HEIGHT = BOARD_HEIGHT + 2 * PANEL_HEIGHT
SQ_SIZE = BOARD_WIDTH // 8
FPS = 60
BOARD_Y_OFFSET = PANEL_HEIGHT
PIECE_IMAGE_NAMES = {
    'P': 'Pawn - W.png', 'N': 'Knight - W.png', 'B': 'Bishop - W.png', 'R': 'Rook - W.png', 'Q': 'Queen - W.png', 'K': 'King - W.png',
    'p': 'Pawn - B.png', 'n': 'Knight - B.png', 'b': 'Bishop - B.png', 'r': 'Rook - B.png', 'q': 'Queen - B.png', 'k': 'King - B.png',
}

PIECE_SYMBOLS = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': ' ',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': ' ', 'k': '♚',
}

# Enhanced color scheme for better readability and contrast
MENU_BG_COLOR = (25, 25, 30)  # Darker for better contrast
MENU_TEXT_COLOR = (250, 250, 255)  # Brighter white for better readability
BUTTON_COLOR = (70, 75, 85)  # Lighter for better visibility
BUTTON_HOVER_COLOR = (90, 95, 105)  # More contrast on hover
BUTTON_TEXT_COLOR = (255, 255, 255)
ACCENT_COLOR = (120, 160, 255)  # Brighter blue for better visibility
SIDE_PANEL_BG = (35, 35, 40)  # Lighter for better text contrast
CAPTURE_AREA_BG = (45, 45, 50)  # Better contrast
SUCCESS_COLOR = (100, 200, 100)  # Brighter green
WARNING_COLOR = (255, 180, 50)  # Better orange for visibility
ERROR_COLOR = (255, 100, 100)  # Brighter red

# Enhanced font system for better readability
def get_fonts():
    """Get optimized fonts for better readability"""
    try:
        # Try to use better fonts if available - slightly larger for better readability
        title_font = pygame.font.SysFont('Arial', 36, bold=True)  # Increased from 32
        heading_font = pygame.font.SysFont('Arial', 26, bold=True)  # Increased from 24
        text_font = pygame.font.SysFont('Arial', 22)  # Increased from 20
        small_font = pygame.font.SysFont('Arial', 18)  # Increased from 16
        button_font = pygame.font.SysFont('Arial', 20, bold=True)  # Increased from 18
    except:
        # Fallback to default fonts
        title_font = pygame.font.SysFont(None, 40)  # Increased
        heading_font = pygame.font.SysFont(None, 30)  # Increased
        text_font = pygame.font.SysFont(None, 26)  # Increased
        small_font = pygame.font.SysFont(None, 22)  # Increased
        button_font = pygame.font.SysFont(None, 24)  # Increased
    
    return {
        'title': title_font,
        'heading': heading_font,
        'text': text_font,
        'small': small_font,
        'button': button_font
    }

def draw_text_with_shadow(surface, font, text, color, x, y, shadow_color=(0, 0, 0), shadow_offset=2):
    """Draw text with enhanced shadow for better readability"""
    # Draw shadow with better offset for readability
    shadow_text = font.render(text, True, shadow_color)
    surface.blit(shadow_text, (x + shadow_offset, y + shadow_offset))
    # Draw main text
    main_text = font.render(text, True, color)
    surface.blit(main_text, (x, y))
    return main_text.get_rect(x=x, y=y)

def load_piece_images():
    images = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pawns_dir = os.path.join(base_dir, 'Pawns')
    for symbol, filename in PIECE_IMAGE_NAMES.items():
        path = os.path.join(pawns_dir, filename)
        try:
            image = pygame.image.load(path)
            image = pygame.transform.smoothscale(image, (SQ_SIZE, SQ_SIZE))
            images[symbol] = image
        except Exception as e:
            images[symbol] = None
    return images

def draw_gradient_rect(surface, color1, color2, rect):
    """Draw a gradient rectangle"""
    for y in range(rect.height):
        ratio = y / rect.height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        pygame.draw.line(surface, (r, g, b), 
                        (rect.x, rect.y + y), (rect.x + rect.width, rect.y + y))

def draw_board(screen, board, images, settings, selected_square=None, highlight_move=None, animating_piece=None):
    board_colors = BOARD_THEMES[settings['board_theme']]
    highlight_color = HIGHLIGHT_COLORS[settings['highlight_color']]
    
    # Board background with subtle shadow
    shadow_rect = pygame.Rect(5, BOARD_Y_OFFSET + 5, BOARD_WIDTH, BOARD_HEIGHT)
    pygame.draw.rect(screen, (20, 20, 20), shadow_rect)
    
    for rank in range(8):
        for file in range(8):
            display_rank = 7 - rank if not settings['board_flipped'] else rank
            display_file = file if not settings['board_flipped'] else 7 - file
            
            color = board_colors[(rank + file) % 2]
            rect = pygame.Rect(file * SQ_SIZE, rank * SQ_SIZE + BOARD_Y_OFFSET, SQ_SIZE, SQ_SIZE)
            
            # Draw square with theme colors
            pygame.draw.rect(screen, color, rect)
            
            sq = chess.square(display_file, display_rank)
            
            # Enhanced square highlighting
            if selected_square == sq:
                pygame.draw.rect(screen, highlight_color, rect, 4)
                overlay = pygame.Surface((SQ_SIZE, SQ_SIZE))
                overlay.set_alpha(50)
                overlay.fill(highlight_color)
                screen.blit(overlay, rect)
            elif highlight_move and (highlight_move.from_square == sq or highlight_move.to_square == sq):
                pygame.draw.rect(screen, ACCENT_COLOR, rect, 3)
                overlay = pygame.Surface((SQ_SIZE, SQ_SIZE))
                overlay.set_alpha(30)
                overlay.fill(ACCENT_COLOR)
                screen.blit(overlay, rect)
            
            # Draw coordinates if enabled with better contrast
            if settings['show_coordinates']:
                coord_font = pygame.font.SysFont('Arial', 14, bold=True)  # Bold for better readability
                coord_shadow_color = (0, 0, 0)  # Black shadow for contrast
                coord_text_color = (255, 255, 255) if (rank + file) % 2 == 0 else (0, 0, 0)  # Contrast with square color
                
                if rank == 7:  # Bottom rank for files
                    file_char = chr(ord('a') + display_file)
                    # Draw shadow for better visibility
                    shadow_text = coord_font.render(file_char, True, coord_shadow_color)
                    screen.blit(shadow_text, (rect.x + 6, rect.y + SQ_SIZE - 14))
                    # Draw main text
                    file_text = coord_font.render(file_char, True, coord_text_color)
                    screen.blit(file_text, (rect.x + 5, rect.y + SQ_SIZE - 15))
                    
                if file == 0:  # Left file for ranks
                    rank_str = str(display_rank + 1)
                    # Draw shadow for better visibility
                    shadow_text = coord_font.render(rank_str, True, coord_shadow_color)
                    screen.blit(shadow_text, (rect.x + SQ_SIZE - 13, rect.y + 6))
                    # Draw main text
                    rank_text = coord_font.render(rank_str, True, coord_text_color)
                    screen.blit(rank_text, (rect.x + SQ_SIZE - 14, rect.y + 5))
            
            # Skip drawing piece if it's being animated
            if animating_piece and animating_piece[0] == sq:
                continue
                
            piece = board.piece_at(sq)
            if piece:
                symbol = piece.symbol()
                if symbol in images and images[symbol] is not None:
                    screen.blit(images[symbol], rect)
                else:
                    # Fallback piece drawing
                    piece_color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                    pygame.draw.circle(screen, piece_color, rect.center, SQ_SIZE // 3)
                    font = pygame.font.SysFont(None, 48)
                    text = font.render(PIECE_SYMBOLS[symbol], True, (255, 0, 0))
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

def get_square_under_mouse(pos, settings):
    x, y = pos
    if x >= BOARD_WIDTH:  # Click is on side panel
        return None
    file = x // SQ_SIZE
    rank = 7 - ((y - BOARD_Y_OFFSET) // SQ_SIZE)
    
    if settings['board_flipped']:
        file = 7 - file
        rank = 7 - rank
        
    if 0 <= file < 8 and 0 <= rank < 8:
        return chess.square(file, rank)
    return None

def get_legal_moves_for_square(board, square):
    legal_moves = []
    for move in board.legal_moves:
        if move.from_square == square:
            legal_moves.append(move.to_square)
    return legal_moves

def draw_possible_moves(screen, board, selected_square, settings):
    if selected_square is not None and settings['show_legal_moves']:
        legal_moves = get_legal_moves_for_square(board, selected_square)
        highlight_color = HIGHLIGHT_COLORS[settings['highlight_color']]
        
        for move_square in legal_moves:
            file = chess.square_file(move_square)
            rank = chess.square_rank(move_square)
            
            if settings['board_flipped']:
                file = 7 - file
                rank = 7 - rank
                
            display_rank = 7 - rank
            center_x = file * SQ_SIZE + SQ_SIZE // 2
            center_y = display_rank * SQ_SIZE + SQ_SIZE // 2 + BOARD_Y_OFFSET
            
            # Enhanced move indicators
            if board.piece_at(move_square):  # Capture indicator
                pygame.draw.circle(screen, highlight_color, (center_x, center_y), SQ_SIZE // 3, 4)
            else:  # Regular move indicator
                pygame.draw.circle(screen, highlight_color, (center_x, center_y), SQ_SIZE // 6)

def animate_move(screen, board, images, settings, move, clock):
    """Animate a piece moving from one square to another"""
    if not settings['animate_moves']:
        return
    
    from_file = chess.square_file(move.from_square)
    from_rank = chess.square_rank(move.from_square)
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    
    if settings['board_flipped']:
        from_file, to_file = 7 - from_file, 7 - to_file
        from_rank, to_rank = 7 - from_rank, 7 - to_rank
    
    from_x = from_file * SQ_SIZE + SQ_SIZE // 2
    from_y = (7 - from_rank) * SQ_SIZE + SQ_SIZE // 2 + BOARD_Y_OFFSET
    to_x = to_file * SQ_SIZE + SQ_SIZE // 2
    to_y = (7 - to_rank) * SQ_SIZE + SQ_SIZE // 2 + BOARD_Y_OFFSET
    
    piece = board.piece_at(move.from_square)
    if not piece:
        return
        
    symbol = piece.symbol()
    piece_image = images.get(symbol)
    
    steps = 10  # Reduced for smoother animation
    for step in range(steps + 1):
        progress = step / steps
        # Smooth easing
        progress = progress * progress * (3.0 - 2.0 * progress)
        current_x = from_x + (to_x - from_x) * progress
        current_y = from_y + (to_y - from_y) * progress
        
        # Clear screen and redraw
        screen.fill(MENU_BG_COLOR)
        
        # Get captured pieces for side panel
        captured_white, captured_black = get_captured_pieces(board)
        
        # Redraw everything except moving piece
        draw_board(screen, board, images, settings, animating_piece=(move.from_square, None))
        draw_side_panel(screen, settings, captured_white, captured_black, [], None, board)  # Pass empty history during animation
        
        # Draw moving piece at current position
        if piece_image:
            piece_rect = piece_image.get_rect(center=(int(current_x), int(current_y)))
            screen.blit(piece_image, piece_rect)
        
        pygame.display.flip()
        clock.tick(30)

def handle_pawn_promotion(board, move, screen, clock, selected_square, images, settings):
    if move.promotion is None and board.piece_at(move.from_square).piece_type == chess.PAWN:
        to_rank = chess.square_rank(move.to_square)
        if (board.turn == chess.WHITE and to_rank == 7) or (board.turn == chess.BLACK and to_rank == 0):
            promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
            if board.turn == chess.WHITE:
                symbols = ['Q', 'R', 'B', 'N']
            else:
                symbols = ['q', 'r', 'b', 'n']
            
            # Enhanced promotion dialog
            dialog_width, dialog_height = 400, 200
            dialog_surface = pygame.Surface((dialog_width, dialog_height))
            dialog_surface.fill((240, 240, 240))
            pygame.draw.rect(dialog_surface, (100, 100, 100), dialog_surface.get_rect(), 3)
            
            # Title
            font_title = pygame.font.SysFont(None, 28)
            title = font_title.render("Choose Promotion Piece", True, (0, 0, 0))
            title_rect = title.get_rect(center=(dialog_width // 2, 25))
            dialog_surface.blit(title, title_rect)
            
            dialog_rect = dialog_surface.get_rect(center=(WIDTH // 2, HEIGHT // 2))
            waiting_for_promotion = True
            
            while waiting_for_promotion:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        mouse_pos = pygame.mouse.get_pos()
                        dialog_mouse_x = mouse_pos[0] - dialog_rect.x
                        dialog_mouse_y = mouse_pos[1] - dialog_rect.y
                        if dialog_surface.get_rect().collidepoint(dialog_mouse_x, dialog_mouse_y):
                            for i, symbol in enumerate(symbols):
                                piece_rect = pygame.Rect(50 + i * 80, 60, 70, 70)
                                if piece_rect.collidepoint(dialog_mouse_x, dialog_mouse_y):
                                    return promotion_pieces[i]
                
                # Draw pieces with enhanced appearance
                for i, symbol in enumerate(symbols):
                    piece_rect = pygame.Rect(50 + i * 80, 60, 70, 70)
                    pygame.draw.rect(dialog_surface, (255, 255, 255), piece_rect)
                    pygame.draw.rect(dialog_surface, (0, 0, 0), piece_rect, 2)
                    
                    if images[symbol] is not None:
                        piece_image = pygame.transform.scale(images[symbol], (60, 60))
                        image_rect = piece_image.get_rect(center=piece_rect.center)
                        dialog_surface.blit(piece_image, image_rect)
                    else:
                        font = pygame.font.SysFont(None, 48)
                        text = font.render(PIECE_SYMBOLS[symbol], True, (0, 0, 0))
                        text_rect = text.get_rect(center=piece_rect.center)
                        dialog_surface.blit(text, text_rect)
                
                screen.fill((0, 0, 0))
                draw_board(screen, board, images, settings, selected_square)
                screen.blit(dialog_surface, dialog_rect)
                pygame.display.flip()
                clock.tick(FPS)
    return None

def draw_captured_pieces(screen, captured_white, captured_black, images, settings):
    """Draw captured pieces in the side panel"""
    if not settings['show_captured_pieces']:
        return
        
    font = pygame.font.SysFont(None, 24)
    
    # White captured pieces
    y_offset = BOARD_Y_OFFSET + 20
    white_title = font.render("Captured by Black:", True, MENU_TEXT_COLOR)
    screen.blit(white_title, (BOARD_WIDTH + 10, y_offset))
    y_offset += 30
    
    x_offset = BOARD_WIDTH + 10
    for piece_type, count in captured_white.items():
        symbol = chess.piece_symbol(piece_type).upper()
        for i in range(count):
            if x_offset + 30 > WIDTH - 10:
                x_offset = BOARD_WIDTH + 10
                y_offset += 35
            
            if symbol in images and images[symbol]:
                piece_img = pygame.transform.scale(images[symbol], (30, 30))
                screen.blit(piece_img, (x_offset, y_offset))
            else:
                pygame.draw.circle(screen, (255, 255, 255), (x_offset + 15, y_offset + 15), 12)
                mini_font = pygame.font.SysFont(None, 20)
                text = mini_font.render(PIECE_SYMBOLS[symbol], True, (0, 0, 0))
                screen.blit(text, (x_offset + 8, y_offset + 8))
            x_offset += 35
    
    # Black captured pieces
    y_offset += 50
    black_title = font.render("Captured by White:", True, MENU_TEXT_COLOR)
    screen.blit(black_title, (BOARD_WIDTH + 10, y_offset))
    y_offset += 30
    
    x_offset = BOARD_WIDTH + 10
    for piece_type, count in captured_black.items():
        symbol = chess.piece_symbol(piece_type).lower()
        for i in range(count):
            if x_offset + 30 > WIDTH - 10:
                x_offset = BOARD_WIDTH + 10
                y_offset += 35
            
            if symbol in images and images[symbol]:
                piece_img = pygame.transform.scale(images[symbol], (30, 30))
                screen.blit(piece_img, (x_offset, y_offset))
            else:
                pygame.draw.circle(screen, (0, 0, 0), (x_offset + 15, y_offset + 15), 12)
                mini_font = pygame.font.SysFont(None, 20)
                text = mini_font.render(PIECE_SYMBOLS[symbol], True, (255, 255, 255))
                screen.blit(text, (x_offset + 8, y_offset + 8))
            x_offset += 35

def draw_move_history(screen, move_history, settings):
    """Draw recent moves in the side panel"""
    if not settings['show_move_history'] or not move_history:
        return
        
    font = pygame.font.SysFont(None, 20)
    title_font = pygame.font.SysFont(None, 24)
    
    y_start = HEIGHT - PANEL_HEIGHT - 200
    title = title_font.render("Recent Moves:", True, MENU_TEXT_COLOR)
    screen.blit(title, (BOARD_WIDTH + 10, y_start))
    
    y_offset = y_start + 30
    recent_moves = move_history[-10:]  # Show last 10 moves
    
    for i, move in enumerate(recent_moves):
        move_num = len(move_history) - len(recent_moves) + i + 1
        if i % 2 == 0:  # White move
            text = f"{(move_num + 1) // 2}. {move}"
        else:  # Black move
            text = f"   {move}"
        
        move_text = font.render(text, True, MENU_TEXT_COLOR)
        screen.blit(move_text, (BOARD_WIDTH + 10, y_offset))
        y_offset += 20

def draw_enhanced_menu(screen, settings, ai_settings, font, mouse_pos):
    """Enhanced settings menu with proper spacing to prevent overlaps"""
    screen.fill(MENU_BG_COLOR)
    fonts = get_fonts()
    
    # Title with better styling - make title area smaller to prevent overlap
    title_rect = pygame.Rect(0, 0, WIDTH, 70)  # Reduced height
    draw_gradient_rect(screen, ACCENT_COLOR, (60, 80, 180), title_rect)
    
    # Main title with shadow - centered properly
    title_text = "⚙️ Game Settings"
    title_width = fonts['title'].size(title_text)[0]
    draw_text_with_shadow(screen, fonts['title'], title_text, 
                         (255, 255, 255), WIDTH//2 - title_width//2, 15)  # Higher position
    
    # Subtitle - positioned to not overlap with content below
    subtitle_text = fonts['small'].render(" ", True, (220, 220, 220))
    subtitle_rect = subtitle_text.get_rect(center=(WIDTH//2, 45))  # Higher position
    screen.blit(subtitle_text, subtitle_rect)
    
    # Main content area - start lower to avoid overlap
    content_y = 70  # Reduced from 80
    col1_x, col2_x = 20, WIDTH // 2 + 20  # Reduced margins
    
    # ===== LEFT COLUMN =====
    y_pos = content_y
    
    # AI Settings Section
    ai_header_rect = pygame.Rect(col1_x - 5, y_pos - 5, min(300, WIDTH//2 - 50), 30)
    pygame.draw.rect(screen, (50, 50, 55), ai_header_rect, border_radius=5)
    
    draw_text_with_shadow(screen, fonts['heading'], "🤖 AI Configuration", 
                         SUCCESS_COLOR, col1_x, y_pos)
    y_pos += 40
    
    # AI Depth with better spacing
    difficulty_names = {1: "Beginner", 2: "Easy", 3: "Normal", 4: "Hard", 5: "Expert", 6: "Master"}
    difficulty_text = difficulty_names.get(ai_settings['search_depth'], "Custom")
    
    draw_enhanced_setting_row(screen, fonts, "Intelligence Level:", 
                             f"{ai_settings['search_depth']} ({difficulty_text})", 
                             col1_x, y_pos, mouse_pos, 'depth', 
                             "Higher = stronger but slower")
    y_pos += 65  # Reduced spacing
    
    # AI Aggressivity
    aggro_desc = {
        0.0: "Defensive", 0.5: "Balanced", 1.0: "Normal", 
        1.5: "Aggressive", 2.0: "Very Aggressive"
    }
    aggro_style = aggro_desc.get(round(ai_settings['aggressivity_factor'], 1), "Custom")
    
    draw_enhanced_setting_row(screen, fonts, "Playing Style:", 
                             f"{ai_settings['aggressivity_factor']:.1f} ({aggro_style})", 
                             col1_x, y_pos, mouse_pos, 'aggro',
                             "AI's tactical approach")
    y_pos += 70  # Reduced spacing
    
    # Visual Settings Section
    visual_header_rect = pygame.Rect(col1_x - 5, y_pos - 5, min(300, WIDTH//2 - 50), 30)
    pygame.draw.rect(screen, (50, 50, 55), visual_header_rect, border_radius=5)
    
    draw_text_with_shadow(screen, fonts['heading'], "🎨 Visual Settings", 
                         WARNING_COLOR, col1_x, y_pos)
    y_pos += 40
    
    # Board Theme
    draw_enhanced_setting_row(screen, fonts, "Board Theme:", settings['board_theme'], 
                             col1_x, y_pos, mouse_pos, 'theme',
                             "Board color scheme")
    y_pos += 65  # Reduced spacing
    
    # Highlight Color
    draw_enhanced_setting_row(screen, fonts, "Highlight Color:", settings['highlight_color'], 
                             col1_x, y_pos, mouse_pos, 'highlight',
                             "Selection highlight style")
    
    # ===== RIGHT COLUMN =====
    y_pos = content_y
    
    # Game Features Section
    features_header_rect = pygame.Rect(col2_x - 5, y_pos - 5, min(300, WIDTH//2 - 50), 30)
    pygame.draw.rect(screen, (50, 50, 55), features_header_rect, border_radius=5)
    
    draw_text_with_shadow(screen, fonts['heading'], "🎮 Game Features", 
                         ACCENT_COLOR, col2_x, y_pos)
    y_pos += 40
    
    # Enhanced toggle settings with much better spacing
    toggle_settings = [
        ('show_legal_moves', 'Show Legal Moves', 'Highlight moves'),
        ('animate_moves', 'Animate Moves', 'Smooth animations'),
        ('show_coordinates', 'Show Coordinates', 'Board labels'),
        ('show_captured_pieces', 'Show Captured', 'Taken pieces'),
        ('show_move_history', 'Move History', 'Recent moves'),
        ('board_flipped', 'Flip Board', 'Rotate view'),
        ('auto_save', 'Auto-Save', 'Save games')
    ]
    
    for setting_key, display_name, description in toggle_settings:
        draw_enhanced_toggle_setting(screen, fonts, display_name, settings[setting_key], 
                                   col2_x, y_pos, mouse_pos, setting_key, description)
        y_pos += 50  # Reduced spacing between toggle items
    
    # Action buttons positioned higher to avoid overlap
    button_y = HEIGHT - 60  # Higher position
    button_width = 120  # Slightly smaller buttons
    button_height = 30
    button_spacing = 30
    
    total_width = 3 * button_width + 2 * button_spacing
    start_x = (WIDTH - total_width) // 2
    
    # Ensure buttons don't go off screen
    if start_x < 10:
        start_x = 10
        button_width = (WIDTH - 100) // 3
        button_spacing = 25
    
    # Reset button
    draw_premium_button(screen, fonts['button'], "🔄 Reset", 
                       start_x, button_y, button_width, button_height, 
                       mouse_pos, 'reset', ERROR_COLOR)
    
    # New Game button
    draw_premium_button(screen, fonts['button'], "🎮 New Game", 
                       start_x + button_width + button_spacing, button_y, 
                       button_width, button_height, mouse_pos, 'new_game', SUCCESS_COLOR)
    
    # Back button
    draw_premium_button(screen, fonts['button'], "← Back", 
                       start_x + 2 * (button_width + button_spacing), button_y, 
                       button_width, button_height, mouse_pos, 'back', ACCENT_COLOR)

def draw_enhanced_setting_row(screen, fonts, label, value, x, y, mouse_pos, setting_type, description):
    """Enhanced setting row with buttons positioned right next to the values"""
    max_width = WIDTH // 2 - 60
    
    # Main label
    if fonts['text'].size(label)[0] > max_width - 100:
        while fonts['text'].size(label + "...")[0] > max_width - 100 and len(label) > 10:
            label = label[:-1]
        label += "..."
    
    draw_text_with_shadow(screen, fonts['text'], label, MENU_TEXT_COLOR, x, y)
    
    # Value with highlighting
    value_str = str(value)
    if fonts['text'].size(value_str)[0] > 100:  # Allow more space for value
        while fonts['text'].size(value_str + "...")[0] > 100 and len(value_str) > 5:
            value_str = value_str[:-1]
        value_str += "..."
    
    value_color = WARNING_COLOR if setting_type in ['depth', 'aggro'] else (150, 200, 255)
    value_x = x + 150
    draw_text_with_shadow(screen, fonts['text'], value_str, value_color, value_x, y)
    
    # Buttons positioned right after the value text
    value_width = fonts['text'].size(value_str)[0]
    button_x = value_x + value_width + 20  # More space for AI configuration buttons
    minus_rect = pygame.Rect(button_x, y - 2, 26, 26)
    plus_rect = pygame.Rect(button_x + 32, y - 2, 26, 26)
    
    # Description text - positioned below and truncated to avoid button area completely
    desc_max_width = max_width - 10  # Use full width but leave margin
    if fonts['small'].size(description)[0] > desc_max_width:
        while fonts['small'].size(description + "...")[0] > desc_max_width and len(description) > 8:
            description = description[:-1]
        description += "..."
    
    desc_text = fonts['small'].render(description, True, (160, 160, 165))
    screen.blit(desc_text, (x, y + 30))  # Move further down to avoid buttons
    
    # Button styling with hover effects
    minus_hover = minus_rect.collidepoint(mouse_pos)
    plus_hover = plus_rect.collidepoint(mouse_pos)
    
    minus_color = (200, 100, 100) if minus_hover else (140, 70, 70)
    plus_color = (100, 200, 100) if plus_hover else (70, 140, 70)
    
    pygame.draw.rect(screen, minus_color, minus_rect, border_radius=5)
    pygame.draw.rect(screen, plus_color, plus_rect, border_radius=5)
    
    # Button text
    minus_text = fonts['small'].render("−", True, (255, 255, 255))
    plus_text = fonts['small'].render("+", True, (255, 255, 255))
    
    minus_text_rect = minus_text.get_rect(center=minus_rect.center)
    plus_text_rect = plus_text.get_rect(center=plus_rect.center)
    
    screen.blit(minus_text, minus_text_rect)
    screen.blit(plus_text, plus_text_rect)

def draw_enhanced_toggle_setting(screen, fonts, label, value, x, y, mouse_pos, setting_key, description):
    """Enhanced toggle setting with switch positioned right next to the label text"""
    max_width = WIDTH // 2 - 60
    
    # Main label
    if fonts['text'].size(label)[0] > max_width - 100:
        while fonts['text'].size(label + "...")[0] > max_width - 100 and len(label) > 10:
            label = label[:-1]
        label += "..."
    
    draw_text_with_shadow(screen, fonts['text'], label, MENU_TEXT_COLOR, x, y)
    
    # Toggle switch positioned right after the label text
    label_width = fonts['text'].size(label)[0]
    switch_x = x + label_width + 15  # Small gap after label text
    switch_rect = pygame.Rect(switch_x, y + 2, 55, 22)
    
    # Description text - positioned below and truncated to use full width
    desc_max_width = max_width - 10  # Use full width but leave margin  
    if fonts['small'].size(description)[0] > desc_max_width:
        while fonts['small'].size(description + "...")[0] > desc_max_width and len(description) > 8:
            description = description[:-1]
        description += "..."
    
    desc_text = fonts['small'].render(description, True, (160, 160, 165))
    screen.blit(desc_text, (x, y + 30))  # Move further down to avoid switch
    
    # Draw the toggle switch
    switch_color = SUCCESS_COLOR if value else (120, 120, 125)
    
    if switch_rect.collidepoint(mouse_pos):
        switch_color = tuple(min(255, c + 30) for c in switch_color)
    
    # Draw switch background
    pygame.draw.rect(screen, switch_color, switch_rect, border_radius=11)
    pygame.draw.rect(screen, (255, 255, 255, 80), switch_rect, 1, border_radius=11)
    
    # Switch knob
    knob_x = switch_x + 32 if value else switch_x + 3
    knob_rect = pygame.Rect(knob_x, y + 4, 18, 18)
    pygame.draw.ellipse(screen, (255, 255, 255), knob_rect)
    pygame.draw.ellipse(screen, (200, 200, 200), knob_rect, 1)
    
    # Status text positioned right after the switch
    status_color = SUCCESS_COLOR if value else (160, 160, 165)
    status_text = fonts['small'].render("ON" if value else "OFF", True, status_color)
    status_x = switch_x + 65
    screen.blit(status_text, (status_x, y + 6))

def draw_premium_button(screen, font, text, x, y, width, height, mouse_pos, button_id, base_color):
    """Draw a premium-styled button with better visual feedback"""
    button_rect = pygame.Rect(x, y, width, height)
    is_hovered = button_rect.collidepoint(mouse_pos)
    
    # Color variations
    if is_hovered:
        bg_color = tuple(min(255, c + 40) for c in base_color)
        border_color = (255, 255, 255)
        text_color = (255, 255, 255)
    else:
        bg_color = base_color
        border_color = tuple(min(255, c + 60) for c in base_color)
        text_color = (240, 240, 240)
    
    # Draw button with gradient effect
    pygame.draw.rect(screen, bg_color, button_rect, border_radius=8)
    
    # Add highlight on top
    highlight_rect = pygame.Rect(x, y, width, height // 3)
    highlight_color = tuple(min(255, c + 20) for c in bg_color)
    pygame.draw.rect(screen, highlight_color, highlight_rect, border_radius=8)
    
    # Border
    pygame.draw.rect(screen, border_color, button_rect, 2, border_radius=8)
    
    # Button text with shadow
    draw_text_with_shadow(screen, font, text, text_color, 
                         x + width//2 - font.size(text)[0]//2, 
                         y + height//2 - font.size(text)[1]//2,
                         (0, 0, 0), 1)

def draw_side_panel(screen, settings, captured_white, captured_black, move_history, ai, board):
    """Enhanced side panel with better spacing and contrast to prevent overlaps"""
    fonts = get_fonts()
    panel_rect = pygame.Rect(BOARD_WIDTH, BOARD_Y_OFFSET, SIDE_PANEL_WIDTH, BOARD_HEIGHT)
    draw_gradient_rect(screen, SIDE_PANEL_BG, (20, 20, 25), panel_rect)  # Better gradient for contrast
    
    # Panel border with accent
    pygame.draw.line(screen, ACCENT_COLOR, (BOARD_WIDTH, BOARD_Y_OFFSET), 
                    (BOARD_WIDTH, BOARD_Y_OFFSET + BOARD_HEIGHT), 3)
    
    # Game status section with better spacing and contrast
    status_y = BOARD_Y_OFFSET + 10
    
    # Status background - enhanced for better readability
    status_bg = pygame.Rect(BOARD_WIDTH + 5, status_y - 5, SIDE_PANEL_WIDTH - 10, 75)  # Slightly taller
    pygame.draw.rect(screen, (50, 50, 55), status_bg, border_radius=8)
    pygame.draw.rect(screen, ACCENT_COLOR, status_bg, 2, border_radius=8)  # Border for definition
    
    if board.is_game_over():
        if board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            status_text = f"🏆 {winner} Wins!"
            status_color = (255, 215, 0)
            subtitle = "Checkmate!"
        else:
            status_text = "🤝 Draw!"
            status_color = (200, 200, 200)
            subtitle = "Game drawn"
    else:
        turn = "White" if board.turn == chess.WHITE else "Black"
        if board.turn == chess.BLACK and ai and ai.thinking:
            status_text = "🤔 AI Thinking"
            status_color = ACCENT_COLOR
            adaptive_depth = ai.get_adaptive_depth(board) if hasattr(ai, 'get_adaptive_depth') else ai.search_depth 
            subtitle = f"Depth {adaptive_depth} • {ai.get_strength_description()}"
        else:
            status_text = f"▶️ {turn}'s Turn"
            status_color = (255, 255, 255) if board.turn == chess.WHITE else (220, 220, 220)
            subtitle = "Make your move!"
    
    # Ensure status text fits
    if fonts['heading'].size(status_text)[0] > SIDE_PANEL_WIDTH - 20:
        while fonts['heading'].size(status_text)[0] > SIDE_PANEL_WIDTH - 20 and len(status_text) > 8:
            status_text = status_text[:-1]
        status_text += "..."
    
    # Main status with better shadow
    draw_text_with_shadow(screen, fonts['heading'], status_text, status_color, 
                         BOARD_WIDTH + 10, status_y + 8, (0, 0, 0), 2)
    
    # Subtitle with better contrast
    if fonts['text'].size(subtitle)[0] > SIDE_PANEL_WIDTH - 20:  # Changed from small to text font
        while fonts['text'].size(subtitle)[0] > SIDE_PANEL_WIDTH - 20 and len(subtitle) > 5:
            subtitle = subtitle[:-1]
        subtitle += "..."
    
    draw_text_with_shadow(screen, fonts['text'], subtitle, (190, 190, 195), 
                         BOARD_WIDTH + 10, status_y + 35, (0, 0, 0), 1)  # Better shadow
    
    # Check warning with enhanced visibility
    if not board.is_game_over() and board.is_check():
        draw_text_with_shadow(screen, fonts['text'], "⚠️ CHECK!", ERROR_COLOR, 
                             BOARD_WIDTH + 10, status_y + 55, (0, 0, 0), 2)
    
    # Calculate available space for captured pieces and move history
    available_start = status_y + 90  # More space after status section
    available_height = BOARD_HEIGHT - 100  # Total available height
    
    # If both features are enabled, split the space
    if settings['show_captured_pieces'] and settings['show_move_history']:
        captured_height = min(available_height // 2, 150)  # Max 150px for captured
        history_height = available_height - captured_height - 20  # Rest for history with padding
        
        draw_enhanced_captured_pieces(screen, fonts, captured_white, captured_black, images, 
                                    settings, available_start, captured_height)
        draw_enhanced_move_history(screen, fonts, move_history, settings, 
                                 available_start + captured_height + 20, history_height)
    
    elif settings['show_captured_pieces']:
        # Give all space to captured pieces
        draw_enhanced_captured_pieces(screen, fonts, captured_white, captured_black, images, 
                                    settings, available_start, available_height)
    
    elif settings['show_move_history']:
        # Give all space to move history  
        draw_enhanced_move_history(screen, fonts, move_history, settings, 
                                 available_start, available_height)

def draw_enhanced_captured_pieces(screen, fonts, captured_white, captured_black, images, settings, start_y, max_height):
    """Enhanced captured pieces with controlled height and better readability"""
    if max_height < 50:  # Not enough space
        return
        
    current_y = start_y
    
    # Section header with better styling
    header_bg = pygame.Rect(BOARD_WIDTH + 5, current_y - 5, SIDE_PANEL_WIDTH - 10, 30)  # Taller header
    pygame.draw.rect(screen, (55, 55, 60), header_bg, border_radius=5)
    pygame.draw.rect(screen, WARNING_COLOR, header_bg, 1, border_radius=5)  # Border for definition
    
    draw_text_with_shadow(screen, fonts['text'], "📦 Captured Pieces", 
                         WARNING_COLOR, BOARD_WIDTH + 10, current_y + 2, (0, 0, 0), 2)
    current_y += 35
    
    remaining_height = max_height - 35
    pieces_per_row = (SIDE_PANEL_WIDTH - 20) // 28  # Slightly more space per piece
    
    # White pieces (limit display if needed) with better styling
    if captured_white and current_y < start_y + max_height - 35:
        # Label with background for better visibility
        label_bg = pygame.Rect(BOARD_WIDTH + 5, current_y - 2, SIDE_PANEL_WIDTH - 10, 22)
        pygame.draw.rect(screen, (45, 45, 50), label_bg, border_radius=3)
        
        draw_text_with_shadow(screen, fonts['text'], "Taken by Black:", (220, 220, 225), 
                             BOARD_WIDTH + 10, current_y, (0, 0, 0), 1)
        current_y += 25
        
        x_offset = BOARD_WIDTH + 10
        piece_count = 0
        
        for piece_type, count in captured_white.items():
            symbol = chess.piece_symbol(piece_type).upper()
            display_count = min(count, 8)  # Limit pieces shown
            
            for i in range(display_count):
                if piece_count >= pieces_per_row:
                    x_offset = BOARD_WIDTH + 10
                    current_y += 28
                    piece_count = 0
                    if current_y > start_y + max_height - 28:
                        break
                
                # Enhanced piece background with border
                piece_bg = pygame.Rect(x_offset - 2, current_y - 2, 26, 26)  # Slightly larger
                pygame.draw.rect(screen, (70, 70, 75), piece_bg, border_radius=3)
                pygame.draw.rect(screen, (100, 100, 105), piece_bg, 1, border_radius=3)
                
                if symbol in images and images[symbol]:
                    piece_img = pygame.transform.scale(images[symbol], (20, 20))
                    screen.blit(piece_img, (x_offset + 1, current_y + 1))
                else:
                    text = fonts['small'].render(PIECE_SYMBOLS[symbol], True, (255, 255, 255))
                    text_rect = text.get_rect(center=(x_offset + 11, current_y + 11))
                    screen.blit(text, text_rect)
                
                x_offset += 28
                piece_count += 1
            
            if current_y > start_y + max_height - 28:
                break
        
        current_y += 35
    
    # Black pieces with better styling
    if captured_black and current_y < start_y + max_height - 35:
        # Label with background for better visibility
        label_bg = pygame.Rect(BOARD_WIDTH + 5, current_y - 2, SIDE_PANEL_WIDTH - 10, 22)
        pygame.draw.rect(screen, (45, 45, 50), label_bg, border_radius=3)
        
        draw_text_with_shadow(screen, fonts['text'], "Taken by White:", (220, 220, 225), 
                             BOARD_WIDTH + 10, current_y, (0, 0, 0), 1)
        current_y += 25
        
        x_offset = BOARD_WIDTH + 10
        piece_count = 0
        
        for piece_type, count in captured_black.items():
            symbol = chess.piece_symbol(piece_type).lower()
            display_count = min(count, 8)  # Limit pieces shown
            
            for i in range(display_count):
                if piece_count >= pieces_per_row:
                    x_offset = BOARD_WIDTH + 10
                    current_y += 28
                    piece_count = 0
                    if current_y > start_y + max_height - 28:
                        break
                
                # Enhanced piece background with border
                piece_bg = pygame.Rect(x_offset - 2, current_y - 2, 26, 26)  # Slightly larger
                pygame.draw.rect(screen, (70, 70, 75), piece_bg, border_radius=3)
                pygame.draw.rect(screen, (100, 100, 105), piece_bg, 1, border_radius=3)
                
                if symbol in images and images[symbol]:
                    piece_img = pygame.transform.scale(images[symbol], (20, 20))
                    screen.blit(piece_img, (x_offset + 1, current_y + 1))
                else:
                    text = fonts['small'].render(PIECE_SYMBOLS[symbol], True, (0, 0, 0))
                    text_rect = text.get_rect(center=(x_offset + 11, current_y + 11))
                    # Add white outline for black pieces for better visibility
                    outline_positions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    for dx, dy in outline_positions:
                        outline_text = fonts['small'].render(PIECE_SYMBOLS[symbol], True, (255, 255, 255))
                        screen.blit(outline_text, (text_rect.x + dx, text_rect.y + dy))
                    screen.blit(text, text_rect)
                
                x_offset += 28
                piece_count += 1
            
            if current_y > start_y + max_height - 28:
                break

def draw_enhanced_move_history(screen, fonts, move_history, settings, start_y, max_height):
    """Enhanced move history with controlled height and better readability"""
    if max_height < 50 or not move_history:
        return
        
    current_y = start_y
    
    # Section header with better styling
    header_bg = pygame.Rect(BOARD_WIDTH + 5, current_y - 5, SIDE_PANEL_WIDTH - 10, 30)  # Taller header
    pygame.draw.rect(screen, (55, 55, 60), header_bg, border_radius=5)
    pygame.draw.rect(screen, ACCENT_COLOR, header_bg, 1, border_radius=5)  # Border for definition
    
    draw_text_with_shadow(screen, fonts['text'], "📝 Move History", 
                         ACCENT_COLOR, BOARD_WIDTH + 10, current_y + 2, (0, 0, 0), 2)
    current_y += 35
    
    # Calculate how many moves we can fit
    remaining_height = max_height - 35
    max_moves = max(1, remaining_height // 25)  # Each move takes ~25px (increased spacing)
    
    recent_moves = move_history[-max_moves:] if len(move_history) > max_moves else move_history
    
    for i, move in enumerate(recent_moves):
        if current_y > start_y + max_height - 25:
            break
            
        move_num = len(move_history) - len(recent_moves) + i + 1
        
        # Enhanced alternating background
        move_bg = pygame.Rect(BOARD_WIDTH + 5, current_y - 3, SIDE_PANEL_WIDTH - 10, 24)
        if i % 2 == 0:
            pygame.draw.rect(screen, (50, 50, 55), move_bg, border_radius=4)
        else:
            pygame.draw.rect(screen, (45, 45, 50), move_bg, border_radius=4)
        
        # Add subtle border for definition
        pygame.draw.rect(screen, (70, 70, 75), move_bg, 1, border_radius=4)
        
        if i % 2 == 0:  # White move
            move_text = f"{(move_num + 1) // 2}. {move}"
            text_color = (255, 255, 255)
            shadow_color = (0, 0, 0)
        else:  # Black move
            move_text = f"   ...{move}"  # Better formatting for black moves
            text_color = (220, 220, 220)
            shadow_color = (0, 0, 0)
        
        # Truncate if too long with better handling
        max_chars = 12  # Slightly more characters
        if len(move_text) > max_chars:
            move_text = move_text[:max_chars-2] + ".."
        
        # Draw move text with shadow for better readability
        draw_text_with_shadow(screen, fonts['text'], move_text, text_color, 
                             BOARD_WIDTH + 10, current_y, shadow_color, 1)
        current_y += 25  # Increased spacing

def draw_top_banner(screen, font, message, is_game_over=False):
    """Enhanced top banner with better styling and readability"""
    fonts = get_fonts()
    panel_rect = pygame.Rect(0, 0, WIDTH, PANEL_HEIGHT)
    draw_gradient_rect(screen, (40, 40, 45), MENU_BG_COLOR, panel_rect)  # Better gradient
    pygame.draw.line(screen, ACCENT_COLOR, (0, PANEL_HEIGHT), (WIDTH, PANEL_HEIGHT), 4)

    # Main title with enhanced styling and better shadow
    title_text = f"  {message}  "
    title_width = fonts['title'].size(title_text)[0]
    draw_text_with_shadow(screen, fonts['title'], title_text, 
                         (255, 255, 255), WIDTH//2 - title_width//2, 15, (0, 0, 0), 3)

    # Enhanced subtitle with better contrast
    if is_game_over:
        subtitle = "Press R to play again • M for settings • ESC to quit"
        subtitle_color = WARNING_COLOR
    else:
        subtitle = "Press M for settings • ESC to quit"
        subtitle_color = (220, 220, 225)  # Better contrast
    
    # Better subtitle positioning and shadow
    draw_text_with_shadow(screen, fonts['text'], subtitle, subtitle_color, 
                         WIDTH // 2 - fonts['text'].size(subtitle)[0] // 2, 65, (0, 0, 0), 2)

def draw_bottom_panel(screen, font, board, ai, settings):
    """Enhanced bottom panel with better readability and contrast"""
    fonts = get_fonts()
    panel_rect = pygame.Rect(0, BOARD_HEIGHT + BOARD_Y_OFFSET, WIDTH, PANEL_HEIGHT)
    
    if board.turn == chess.WHITE:
        bg_color1, bg_color2 = (245, 250, 255), (225, 235, 250)  # Better contrast
        text_color = (20, 20, 100)  # Darker for better readability
        turn_message = "Your Turn"
        subtitle = "Select a piece and make your move!"
    else:
        bg_color1, bg_color2 = (40, 60, 60), (20, 20, 20)  # Better contrast
        text_color = (255, 255, 255)
        if ai.thinking:
            adaptive_depth = ai.get_adaptive_depth(board) if hasattr(ai, 'get_adaptive_depth') else ai.search_depth
            turn_message = f"AI Computing..."
            subtitle = f"Analyzing at depth {adaptive_depth} • {ai.get_strength_description()}"
        else:
            turn_message = f"AI's Turn"
            strength_desc = ai.get_strength_description() if hasattr(ai, 'get_strength_description') else f"Depth {ai.search_depth}"
            subtitle = f"Ready to move • {strength_desc}"

    draw_gradient_rect(screen, bg_color1, bg_color2, panel_rect)
    pygame.draw.line(screen, ACCENT_COLOR, (0, BOARD_HEIGHT + BOARD_Y_OFFSET), 
                    (WIDTH, BOARD_HEIGHT + BOARD_Y_OFFSET), 4)

    # Main turn message with subtle shadow for better readability
    turn_width = fonts['heading'].size(turn_message)[0]
    draw_text_with_shadow(screen, fonts['heading'], turn_message, text_color, 
                         WIDTH//2 - turn_width//2, 
                         BOARD_HEIGHT + BOARD_Y_OFFSET + 12, (0, 0, 0), 1)
    
    # Subtitle with better contrast and shadow
    subtitle_color = tuple(max(0, c - 60) for c in text_color)  # Darker version of text color
    subtitle_width = fonts['text'].size(subtitle)[0]
    draw_text_with_shadow(screen, fonts['text'], subtitle, subtitle_color,
                         WIDTH//2 - subtitle_width//2, 
                         BOARD_HEIGHT + BOARD_Y_OFFSET + 45, (0, 0, 0), 2)
    
    # Check warning with enhanced visibility
    if board.is_check() and not board.is_game_over():
        check_text = "⚠️ CHECK!"
        check_width = fonts['text'].size(check_text)[0]
        draw_text_with_shadow(screen, fonts['text'], check_text, ERROR_COLOR, 
                             WIDTH//2 - check_width//2, 
                             BOARD_HEIGHT + BOARD_Y_OFFSET + 70, (0, 0, 0), 3)

def get_captured_pieces(board):
    """Calculate captured pieces"""
    captured_white = Counter()
    captured_black = Counter()
    
    # Count pieces on board
    on_board_white = Counter()
    on_board_black = Counter()
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.color == chess.WHITE:
                on_board_white[piece.piece_type] += 1
            else:
                on_board_black[piece.piece_type] += 1
    
    # Starting pieces
    starting_pieces = {
        chess.PAWN: 8, chess.ROOK: 2, chess.KNIGHT: 2, 
        chess.BISHOP: 2, chess.QUEEN: 1, chess.KING: 1
    }
    
    # Calculate captured
    for piece_type in starting_pieces:
        captured_white[piece_type] = starting_pieces[piece_type] - on_board_white.get(piece_type, 0)
        captured_black[piece_type] = starting_pieces[piece_type] - on_board_black.get(piece_type, 0)
    
    # Remove zero counts
    captured_white = {k: v for k, v in captured_white.items() if v > 0}
    captured_black = {k: v for k, v in captured_black.items() if v > 0}
    
    return captured_white, captured_black

def reset_game(board, ai, player_color, settings):
    board.reset()
    ai.current_game_moves = []
    return board, None, False, player_color, None, []

def handle_settings_click(mouse_x, mouse_y, settings, ai_settings):
    """Handle clicks with buttons positioned right next to their values"""
    fonts = get_fonts()
    col1_x = 25
    max_width = WIDTH // 2 - 60
    
    # AI Depth buttons - calculate position based on value text
    difficulty_names = {1: "Beginner", 2: "Easy", 3: "Normal", 4: "Hard", 5: "Expert", 6: "Master"}
    difficulty_text = difficulty_names.get(ai_settings['search_depth'], "Custom")
    depth_value = f"{ai_settings['search_depth']} ({difficulty_text})"
    depth_value_width = fonts['text'].size(depth_value)[0]
    depth_button_x = col1_x + 150 + depth_value_width + 20
    
    # Make click areas larger and more forgiving - match visual button positions exactly
    if pygame.Rect(depth_button_x - 5, 108, 36, 32).collidepoint(mouse_x, mouse_y):  # Depth -
        ai_settings['search_depth'] = max(1, ai_settings['search_depth'] - 1)
        return None
    elif pygame.Rect(depth_button_x + 27, 108, 36, 32).collidepoint(mouse_x, mouse_y):  # Depth +
        ai_settings['search_depth'] = min(6, ai_settings['search_depth'] + 1)
        return None
    
    # AI Aggressivity buttons - calculate position based on value text
    aggro_desc = {0.0: "Defensive", 0.5: "Balanced", 1.0: "Normal", 1.5: "Aggressive", 2.0: "Very Aggressive"}
    aggro_style = aggro_desc.get(round(ai_settings['aggressivity_factor'], 1), "Custom")
    aggro_value = f"{ai_settings['aggressivity_factor']:.1f} ({aggro_style})"
    aggro_value_width = fonts['text'].size(aggro_value)[0]
    aggro_button_x = col1_x + 150 + aggro_value_width + 20
    # Make click areas larger and more forgiving - match visual button positions exactly
    if pygame.Rect(aggro_button_x - 5, 173, 36, 32).collidepoint(mouse_x, mouse_y):  # Aggro -
        ai_settings['aggressivity_factor'] = max(0.0, round(ai_settings['aggressivity_factor'] - 0.1, 1))
        return None
    elif pygame.Rect(aggro_button_x + 27, 173, 36, 32).collidepoint(mouse_x, mouse_y):  # Aggro +
        ai_settings['aggressivity_factor'] = min(2.0, round(ai_settings['aggressivity_factor'] + 0.1, 1))
        return None
    
    # Board Theme buttons - calculate position based on value text
    theme_value_width = fonts['text'].size(settings['board_theme'])[0]
    theme_button_x = col1_x + 150 + theme_value_width + 20
    
    # Make click areas larger and more forgiving - match visual button positions exactly
    if pygame.Rect(theme_button_x - 5, 298, 36, 32).collidepoint(mouse_x, mouse_y):  # Theme -
        themes = list(BOARD_THEMES.keys())
        current_idx = themes.index(settings['board_theme'])
        settings['board_theme'] = themes[(current_idx - 1) % len(themes)]
        return None
    elif pygame.Rect(theme_button_x + 27, 298, 36, 32).collidepoint(mouse_x, mouse_y):  # Theme +
        themes = list(BOARD_THEMES.keys())
        current_idx = themes.index(settings['board_theme'])
        settings['board_theme'] = themes[(current_idx + 1) % len(themes)]
        return None
    
    # Highlight Color buttons - calculate position based on value text
    highlight_value_width = fonts['text'].size(settings['highlight_color'])[0]
    highlight_button_x = col1_x + 150 + highlight_value_width + 20
    
    # Make click areas larger and more forgiving - match visual button positions exactly
    if pygame.Rect(highlight_button_x - 5, 363, 36, 32).collidepoint(mouse_x, mouse_y):  # Highlight -
        colors = list(HIGHLIGHT_COLORS.keys())
        current_idx = colors.index(settings['highlight_color'])
        settings['highlight_color'] = colors[(current_idx - 1) % len(colors)]
        return None
    elif pygame.Rect(highlight_button_x + 27, 363, 36, 32).collidepoint(mouse_x, mouse_y):  # Highlight +
        colors = list(HIGHLIGHT_COLORS.keys())
        current_idx = colors.index(settings['highlight_color'])
        settings['highlight_color'] = colors[(current_idx + 1) % len(colors)]
        return None
    
    # Toggle settings (Column 2) - calculate positions based on label text
    col2_x = WIDTH // 2 + 20
    
    # Updated positions with reduced spacing for smaller screen
    toggle_positions = [110, 160, 210, 260, 310, 360, 410]
    toggle_keys = ['show_legal_moves', 'animate_moves', 'show_coordinates', 
                   'show_captured_pieces', 'show_move_history', 'board_flipped', 'auto_save']
    toggle_labels = ['Show Legal Moves', 'Animate Moves', 'Show Coordinates', 
                     'Show Captured', 'Move History', 'Flip Board', 'Auto-Save']
    
    for i, (setting_key, label, y_pos) in enumerate(zip(toggle_keys, toggle_labels, toggle_positions)):
        label_width = fonts['text'].size(label)[0]
        switch_x = col2_x + label_width + 15
        if pygame.Rect(switch_x, y_pos + 2, 55, 22).collidepoint(mouse_x, mouse_y):
            settings[setting_key] = not settings[setting_key]
            return None
    
    # Action buttons
    button_y = HEIGHT - 70
    button_width = 130
    button_spacing = 35
    
    total_width = 3 * button_width + 2 * button_spacing
    start_x = (WIDTH - total_width) // 2
    
    if start_x < 10:
        start_x = 10
        button_width = (WIDTH - 100) // 3
        button_spacing = 25
    
    if pygame.Rect(start_x, button_y, button_width, 32).collidepoint(mouse_x, mouse_y):
        return 'reset'
    elif pygame.Rect(start_x + button_width + button_spacing, button_y, button_width, 32).collidepoint(mouse_x, mouse_y):
        return 'new_game'  
    elif pygame.Rect(start_x + 2 * (button_width + button_spacing), button_y, button_width, 32).collidepoint(mouse_x, mouse_y):
        return 'back'
    
    return None

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(base_dir, 'Chess.png')
    try:
        if os.path.exists(icon_path):
            icon_surface = pygame.image.load(icon_path)
            pygame.display.set_icon(icon_surface)
    except pygame.error:
        pass

    pygame.display.set_caption('  ChesslerAI   ')
    clock = pygame.time.Clock()
    board = chess.Board()
    player_model = PlayerModel()

    # Initialize settings
    settings = GAME_SETTINGS.copy()
    ai_settings = {
        'search_depth': 3,
        'aggressivity_factor': 1.0,
    }

    ai = AdvancedAdaptiveChessAI(player_model,
                                 search_depth=ai_settings['search_depth'],
                                 aggressivity_factor=ai_settings['aggressivity_factor'])

    selected_square = None
    running = True
    game_recorded = False
    status_font = pygame.font.SysFont(None, 28)
    player_color = chess.WHITE
    global images
    images = load_piece_images()
    move_history = []

    game_state = "playing"
    last_ai_move = None

    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if not game_recorded and len(ai.current_game_moves) > 0:
                    ai.end_game(completed=False)
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if not game_recorded and len(ai.current_game_moves) > 0:
                        ai.end_game(completed=False)
                    running = False
                elif event.key == pygame.K_m:
                    game_state = "menu" if game_state == "playing" else "playing"
                elif event.key == pygame.K_r and board.is_game_over():
                    board, selected_square, game_recorded, player_color, last_ai_move, move_history = reset_game(board, ai, player_color, settings)

            if game_state == "playing" and not board.is_game_over():
                if event.type == pygame.MOUSEBUTTONDOWN and board.turn == player_color:
                    last_ai_move = None
                    sq = get_square_under_mouse(event.pos, settings)
                    if selected_square is None:
                        if sq is not None and board.piece_at(sq) and board.piece_at(sq).color == player_color:
                            selected_square = sq
                    else:
                        if sq is not None:
                            piece = board.piece_at(selected_square)
                            is_promotion_attempt = (
                                piece is not None and
                                piece.piece_type == chess.PAWN and
                                chess.square_rank(sq) in [0, 7] and
                                piece.color == board.turn
                            )

                            potential_move = chess.Move(selected_square, sq, chess.QUEEN)

                            if is_promotion_attempt and potential_move in board.legal_moves:
                                move_for_handler = chess.Move(selected_square, sq)
                                promotion_choice = handle_pawn_promotion(board, move_for_handler, screen, clock, selected_square, images, settings)

                                if promotion_choice:
                                    final_move = chess.Move(selected_square, sq, promotion_choice)
                                    if final_move in board.legal_moves:
                                        if settings['animate_moves']:
                                            animate_move(screen, board, images, settings, final_move, clock)
                                        board.push(final_move)
                                        move_history.append(final_move.uci())
                                        ai.record_player_move(final_move.uci())
                                        if not board.is_game_over() and board.turn != player_color:
                                            ai.start_thinking(board)
                                selected_square = None
                            else:
                                move = chess.Move(selected_square, sq)
                                if move in board.legal_moves:
                                    if settings['animate_moves']:
                                        animate_move(screen, board, images, settings, move, clock)
                                    board.push(move)
                                    move_history.append(move.uci())
                                    ai.record_player_move(move.uci())
                                    if not board.is_game_over() and board.turn != player_color:
                                        ai.start_thinking(board)
                                    selected_square = None
                                else:
                                    if board.piece_at(sq) and board.piece_at(sq).color == player_color:
                                        selected_square = sq
                                    else:
                                        selected_square = None

            elif game_state == "menu":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    action = handle_settings_click(mouse_x, mouse_y, settings, ai_settings)
                    
                    if action == 'reset':
                        settings.update(GAME_SETTINGS)
                        ai_settings.update({'search_depth': 3, 'aggressivity_factor': 1.0})
                    elif action == 'new_game':
                        board, selected_square, game_recorded, player_color, last_ai_move, move_history = reset_game(board, ai, player_color, settings)
                        ai.search_depth = ai_settings['search_depth']
                        ai.aggressivity_factor = ai_settings['aggressivity_factor']
                        game_state = "playing"
                    elif action == 'back':
                        ai.search_depth = ai_settings['search_depth']
                        ai.aggressivity_factor = ai_settings['aggressivity_factor']
                        game_state = "playing"

        if game_state == "playing":
            screen.fill(MENU_BG_COLOR)
            
            # Get captured pieces
            captured_white, captured_black = get_captured_pieces(board)
            
            draw_board(screen, board, images, settings, selected_square, highlight_move=last_ai_move)
            draw_possible_moves(screen, board, selected_square, settings)
            draw_side_panel(screen, settings, captured_white, captured_black, move_history, ai, board)

            if board.is_game_over():
                if not game_recorded:
                    ai.end_game(completed=True)
                    game_recorded = True

                message = ""
                if board.is_checkmate():
                    if board.turn == player_color:
                        message = "Game Over - You Lost!"
                    else:
                        message = "Congratulations - You Won!"
                elif board.is_stalemate() or board.is_insufficient_material():
                    message = "Game Draw - Stalemate!"
                else:
                    message = f"Game Over - {board.result()}"

                draw_top_banner(screen, status_font, message, is_game_over=True)
            else:
                if board.turn != player_color and not ai.thinking:
                    ai_move = ai.get_move()
                    if ai_move:
                        if settings['animate_moves']:
                            animate_move(screen, board, images, settings, ai_move, clock)
                        board.push(ai_move)
                        move_history.append(ai_move.uci())
                        last_ai_move = ai_move
                        if settings['move_delay'] > 0:
                            pygame.time.wait(int(settings['move_delay'] * 1000))
                    else:
                        ai.start_thinking(board)

                draw_top_banner(screen, status_font, "ChesslerAI")

            draw_bottom_panel(screen, status_font, board, ai, settings)

        elif game_state == "menu":
            draw_enhanced_menu(screen, settings, ai_settings, status_font, mouse_pos)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()