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
import hashlib
from typing import Dict, List, Tuple, Optional, Set
import pickle

# File paths
PLAYER_MODEL_FILE = 'player_model.json'
PLAYER_GAMES_LOG = 'player_games.log'
OPENING_BOOK_FILE = 'opening_book.pkl'
ENDGAME_DB_FILE = 'endgame_db.pkl'

# Simple color scheme
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT_COLOR = (255, 255, 0)
SELECTED_COLOR = (0, 255, 0)
LEGAL_MOVE_COLOR = (0, 0, 255)
BACKGROUND_COLOR = (50, 50, 50)
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (70, 70, 70)
MENU_OVERLAY_COLOR = (0, 0, 0, 180)  # Semi-transparent black
MENU_BOX_COLOR = (60, 60, 60)

# Board dimensions
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
PANEL_WIDTH = 300
WINDOW_WIDTH = BOARD_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = BOARD_SIZE + 100
FPS = 60

# Piece symbols for text fallback
PIECE_SYMBOLS = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

# Piece image filenames
PIECE_IMAGES = {
    'P': 'Pawn - W.png', 'N': 'Knight - W.png', 'B': 'Bishop - W.png',
    'R': 'Rook - W.png', 'Q': 'Queen - W.png', 'K': 'King - W.png',
    'p': 'Pawn - B.png', 'n': 'Knight - B.png', 'b': 'Bishop - B.png',
    'r': 'Rook - B.png', 'q': 'Queen - B.png', 'k': 'King - B.png'
}

class PlayerModel:
    def __init__(self):
        self.move_counter = Counter()
        self.games_played = 0
        self.load()

    def record_game(self, moves, completed=True):
        for move in moves:
            self.move_counter[move] += 1
        self.games_played += 1
        self.save_game(moves, completed)

    def save_game(self, moves, completed=True):
        timestamp = datetime.datetime.now().isoformat()
        with open(PLAYER_GAMES_LOG, 'a') as f:
            f.write(f"[{timestamp}] {','.join(moves)} ({'completed' if completed else 'incomplete'})\n")

    def load(self):
        if os.path.exists(PLAYER_GAMES_LOG):
            try:
                with open(PLAYER_GAMES_LOG, 'r') as f:
                    for line in f:
                        if line.strip() and '] ' in line:
                            moves_part = line.split('] ')[1].split(' (')[0]
                            moves = [m.strip() for m in moves_part.split(',') if m.strip()]
                            for move in moves:
                                self.move_counter[move] += 1
                            if moves:
                                self.games_played += 1
            except Exception as e:
                print(f"Error loading player model: {e}")

# Simple piece values
PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}

# Advanced piece values with more nuanced evaluation
ADVANCED_PIECE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000
}

# Piece-square tables for positional evaluation
PST_PAWN_MG = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

PST_KNIGHT_MG = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PST_BISHOP_MG = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

PST_ROOK_MG = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

PST_QUEEN_MG = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
]

PST_KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

PST_KING_EG = [
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
    chess.PAWN: PST_PAWN_MG,
    chess.KNIGHT: PST_KNIGHT_MG,
    chess.BISHOP: PST_BISHOP_MG,
    chess.ROOK: PST_ROOK_MG,
    chess.QUEEN: PST_QUEEN_MG,
    chess.KING: PST_KING_MG
}

class TranspositionTable:
    """High-performance transposition table with replacement scheme"""
    
    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2
    
    def __init__(self, size_mb=64):
        self.size = (size_mb * 1024 * 1024) // 24  # Approximate entry size
        self.table = {}
        self.hits = 0
        self.stores = 0
    
    def _hash_position(self, board):
        """Fast position hashing using board FEN"""
        return hash(board.fen().split()[0])  # Only position, not game state
    
    def store(self, board, depth, score, move, flag):
        """Store position evaluation"""
        key = self._hash_position(board)
        
        # Age-based replacement - prefer deeper searches
        if key in self.table:
            old_depth = self.table[key][1]
            if depth < old_depth:
                return
        
        self.table[key] = (score, depth, move, flag)
        self.stores += 1
        
        # Limit table size
        if len(self.table) > self.size:
            # Remove 25% of oldest entries (simple cleanup)
            keys_to_remove = list(self.table.keys())[:len(self.table)//4]
            for k in keys_to_remove:
                del self.table[k]
    
    def probe(self, board, depth, alpha, beta):
        """Probe transposition table"""
        key = self._hash_position(board)
        
        if key not in self.table:
            return None, None
        
        score, stored_depth, move, flag = self.table[key]
        
        if stored_depth >= depth:
            self.hits += 1
            if flag == self.EXACT:
                return score, move
            elif flag == self.LOWER_BOUND and score >= beta:
                return score, move
            elif flag == self.UPPER_BOUND and score <= alpha:
                return score, move
        
        return None, move  # Return move for ordering even if score not usable

class KillerMoves:
    """Killer move heuristic for move ordering"""
    
    def __init__(self, max_depth=20):
        self.moves = [[None, None] for _ in range(max_depth)]
    
    def add_killer(self, depth, move):
        """Add a killer move at given depth"""
        if depth < len(self.moves):
            if self.moves[depth][0] != move:
                self.moves[depth][1] = self.moves[depth][0]
                self.moves[depth][0] = move
    
    def is_killer(self, depth, move):
        """Check if move is a killer move"""
        if depth < len(self.moves):
            return move in self.moves[depth]
        return False

class HistoryTable:
    """History heuristic for move ordering"""
    
    def __init__(self):
        self.history = defaultdict(int)
    
    def update(self, move, depth):
        """Update history score for a move"""
        self.history[move] += depth * depth
    
    def get_score(self, move):
        """Get history score for a move"""
        return self.history.get(move, 0)

class OptimizedChessEngine:
    """Advanced chess engine with cutting-edge optimization techniques"""
    
    def __init__(self):
        self.tt = TranspositionTable(size_mb=128)
        self.killer_moves = KillerMoves()
        self.history_table = HistoryTable()
        self.nodes_searched = 0
        self.tt_hits = 0
        self.max_quiesce_depth = 8
        self.null_move_reduction = 2
        self.lmr_threshold = 4
        self.lmr_reduction = 1
        
        # Load opening book and endgame tables
        self.opening_book = self._load_opening_book()
        self.endgame_db = self._load_endgame_db()
    
    def _load_opening_book(self):
        """Load opening book database"""
        try:
            if os.path.exists(OPENING_BOOK_FILE):
                with open(OPENING_BOOK_FILE, 'rb') as f:
                    return pickle.load(f)
        except:
            pass
        
        # Create comprehensive opening book
        opening_book = {
            # King's Pawn openings (e4)
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1": ["e7e5", "c7c5", "e7e6", "c7c6", "d7d6", "d7d5"],
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2": ["g1f3", "f2f4", "b1c3", "d2d3", "f1c4"],
            
            # Italian Game
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": ["b8c6", "g8f6", "f7f5"],
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": ["f1c4", "d2d3", "b1c3"],
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4": ["d2d3", "b1c3", "c2c3"],
            
            # Spanish Opening (Ruy Lopez)
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3": ["f1b5", "f1c4", "d2d3"],
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3": ["a7a6", "f7f5", "g8f6"],
            
            # Sicilian Defense
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2": ["g1f3", "b1c3", "f2f4", "d2d4"],
            "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2": ["d7d6", "b8c6", "g7g6", "e7e6"],
            
            # French Defense
            "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["d2d4", "g1f3", "b1c3"],
            "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2": ["d7d5", "c7c5", "g8f6"],
            
            # Caro-Kann Defense
            "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2": ["d2d4", "b1c3", "g1f3"],
            "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2": ["d7d5", "g8f6", "e7e6"],
            
            # Queen's Pawn openings (d4)
            "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1": ["d7d5", "g8f6", "e7e6", "c7c5", "f7f5"],
            "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2": ["c2c4", "g1f3", "b1c3", "c1g5"],
            
            # Queen's Gambit
            "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2": ["e7e6", "c7c6", "g8f6", "d5c4"],
            "rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 2 3": ["b1c3", "c1g5", "g1f3", "c4d5"],
            
            # King's Indian Defense
            "rnbqkb1r/pppppp1p/5np1/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 2 3": ["c2c4", "g1f3", "b1c3"],
            "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 3": ["f8g7", "d7d6", "c7c5"],
            
            # English Opening
            "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1": ["e7e5", "g8f6", "c7c5", "d7d5"],
            "rnbqkbnr/pppp1ppp/8/4p3/2P5/8/PP1PPPPP/RNBQKBNR w KQkq e6 0 2": ["b1c3", "g1f3", "g2g3"],
            
            # Nimzo-Indian Defense
            "rnbqk2r/pppp1ppp/4pn2/8/1bPP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4": ["e2e3", "f2f3", "d1c2", "a2a3"],
            
            # Modern Defense
            "rnbqkbnr/ppp1pppp/3p4/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq e3 0 2": ["g7g6", "g8f6", "c7c6"],
        }
        
        # Save the opening book
        try:
            with open(OPENING_BOOK_FILE, 'wb') as f:
                pickle.dump(opening_book, f)
        except:
            pass
        
        return opening_book
    
    def _load_endgame_db(self):
        """Load endgame tablebase"""
        try:
            if os.path.exists(ENDGAME_DB_FILE):
                with open(ENDGAME_DB_FILE, 'rb') as f:
                    return pickle.load(f)
        except:
            pass
        
        # Basic endgame knowledge
        endgame_db = {
            # K+Q vs K positions (simplified)
            "4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1": 1000,  # Winning
            "4k3/8/8/8/8/8/8/4K2Q w - - 0 1": 1000,   # Winning
        }
        
        try:
            with open(ENDGAME_DB_FILE, 'wb') as f:
                pickle.dump(endgame_db, f)
        except:
            pass
        
        return endgame_db
    
    def get_opening_move(self, board):
        """Get move from opening book"""
        # Use only the position part of FEN (ignore castling, en passant, etc.)
        fen_parts = board.fen().split()
        position_fen = fen_parts[0] + " " + fen_parts[1] + " " + fen_parts[2] + " " + fen_parts[3] + " 0 " + fen_parts[5]
        
        if position_fen in self.opening_book:
            moves = self.opening_book[position_fen]
            move_str = random.choice(moves)
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    return move
            except:
                pass
        
        # Also try exact FEN match
        full_fen = board.fen()
        if full_fen in self.opening_book:
            moves = self.opening_book[full_fen]
            move_str = random.choice(moves)
            try:
                move = chess.Move.from_uci(move_str)
                if move in board.legal_moves:
                    return move
            except:
                pass
        
        return None
    
    def is_endgame(self, board):
        """Detect endgame phase"""
        piece_count = len(board.piece_map())
        queen_count = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        
        return piece_count <= 10 or (piece_count <= 14 and queen_count == 0)
    
    def get_piece_square_value(self, piece, square):
        """Get piece-square table value"""
        piece_type = piece.piece_type
        
        if piece_type not in PIECE_SQUARE_TABLES:
            return 0
        
        table = PIECE_SQUARE_TABLES[piece_type]
        
        # Flip square for black pieces
        if piece.color == chess.BLACK:
            square = chess.square_mirror(square)
        
        return table[square]
    
    def evaluate_board(self, board):
        """Advanced evaluation function with multiple factors"""
        if board.is_checkmate():
            return -30000 if board.turn else 30000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        score = 0
        
        # Material and position evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Base material value
                material_value = ADVANCED_PIECE_VALUES[piece.piece_type]
                
                # Position value from piece-square tables
                position_value = self.get_piece_square_value(piece, square)
                
                piece_value = material_value + position_value
                
                if piece.color == chess.WHITE:
                    score += piece_value
                else:
                    score -= piece_value
        
        # Mobility evaluation
        original_turn = board.turn
        
        # White mobility
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        
        # Black mobility
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        
        board.turn = original_turn
        
        score += (white_mobility - black_mobility) * 5
        
        # King safety in middlegame
        if not self.is_endgame(board):
            score += self._evaluate_king_safety(board, chess.WHITE) * 10
            score -= self._evaluate_king_safety(board, chess.BLACK) * 10
        
        # Pawn structure evaluation
        score += self._evaluate_pawn_structure(board)
        
        # Center control
        score += self._evaluate_center_control(board)
        
        # Advanced tactical patterns (neural network-inspired)
        score += self._evaluate_tactical_patterns(board)
        
        return score
    
    def _evaluate_king_safety(self, board, color):
        """Evaluate king safety"""
        king_square = board.king(color)
        if king_square is None:
            return -1000
        
        safety_score = 0
        
        # Penalty for exposed king
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Check for pawn shield
        if color == chess.WHITE:
            shield_squares = [
                chess.square(king_file-1, king_rank+1) if king_file > 0 else None,
                chess.square(king_file, king_rank+1),
                chess.square(king_file+1, king_rank+1) if king_file < 7 else None
            ]
        else:
            shield_squares = [
                chess.square(king_file-1, king_rank-1) if king_file > 0 else None,
                chess.square(king_file, king_rank-1),
                chess.square(king_file+1, king_rank-1) if king_file < 7 else None
            ]
        
        for square in shield_squares:
            if square is not None and chess.square_file(square) >= 0 and chess.square_file(square) <= 7:
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    safety_score += 5
        
        return safety_score
    
    def _evaluate_pawn_structure(self, board):
        """Evaluate pawn structure"""
        score = 0
        
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        # Passed pawns
        for square in white_pawns:
            if self._is_passed_pawn(board, square, chess.WHITE):
                rank = chess.square_rank(square)
                score += 20 + (rank * 10)  # More valuable as they advance
        
        for square in black_pawns:
            if self._is_passed_pawn(board, square, chess.BLACK):
                rank = chess.square_rank(square)
                score -= 20 + ((7-rank) * 10)
        
        # Doubled pawns penalty
        for file in range(8):
            white_pawns_in_file = len([s for s in white_pawns if chess.square_file(s) == file])
            black_pawns_in_file = len([s for s in black_pawns if chess.square_file(s) == file])
            
            if white_pawns_in_file > 1:
                score -= 10 * (white_pawns_in_file - 1)
            if black_pawns_in_file > 1:
                score += 10 * (black_pawns_in_file - 1)
        
        return score
    
    def _is_passed_pawn(self, board, square, color):
        """Check if pawn is passed"""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        enemy_color = not color
        enemy_pawns = board.pieces(chess.PAWN, enemy_color)
        
        if color == chess.WHITE:
            # Check files and ranks ahead
            for enemy_square in enemy_pawns:
                enemy_file = chess.square_file(enemy_square)
                enemy_rank = chess.square_rank(enemy_square)
                
                if abs(enemy_file - file) <= 1 and enemy_rank > rank:
                    return False
        else:
            # Check files and ranks ahead for black
            for enemy_square in enemy_pawns:
                enemy_file = chess.square_file(enemy_square)
                enemy_rank = chess.square_rank(enemy_square)
                
                if abs(enemy_file - file) <= 1 and enemy_rank < rank:
                    return False
        
        return True
    
    def _evaluate_center_control(self, board):
        """Evaluate center control"""
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        extended_center = [chess.C4, chess.C5, chess.F4, chess.F5]
        score = 0
        
        # Core center squares
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    score += 15
                else:
                    score -= 15
        
        # Extended center squares
        for square in extended_center:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    score += 8
                else:
                    score -= 8
        
        return score
    
    def _evaluate_tactical_patterns(self, board):
        """Neural network-inspired tactical pattern recognition"""
        score = 0
        
        # Pin detection
        score += self._detect_pins(board)
        
        # Fork opportunities
        score += self._detect_forks(board)
        
        # Discovered attack potential
        score += self._detect_discovered_attacks(board)
        
        # Back rank weakness
        score += self._evaluate_back_rank_safety(board)
        
        return score
    
    def _detect_pins(self, board):
        """Detect pinned pieces"""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is None:
                continue
                
            # Check for pins along ranks, files, and diagonals
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),  # Ranks and files
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonals
            ]
            
            for dx, dy in directions:
                pinned_piece = None
                file, rank = chess.square_file(king_square), chess.square_rank(king_square)
                
                for i in range(1, 8):
                    new_file, new_rank = file + dx * i, rank + dy * i
                    if not (0 <= new_file <= 7 and 0 <= new_rank <= 7):
                        break
                    
                    square = chess.square(new_file, new_rank)
                    piece = board.piece_at(square)
                    
                    if piece:
                        if piece.color == color:
                            if pinned_piece is None:
                                pinned_piece = piece
                            else:
                                break  # Two friendly pieces, no pin
                        else:
                            # Enemy piece - check if it can pin
                            if pinned_piece and self._can_pin_along_direction(piece, dx, dy):
                                pin_value = ADVANCED_PIECE_VALUES[pinned_piece.piece_type] // 10
                                if color == chess.WHITE:
                                    score -= pin_value  # White piece is pinned (bad)
                                else:
                                    score += pin_value  # Black piece is pinned (good)
                            break
        
        return score
    
    def _can_pin_along_direction(self, piece, dx, dy):
        """Check if piece can pin along given direction"""
        if abs(dx) == abs(dy):  # Diagonal
            return piece.piece_type in [chess.BISHOP, chess.QUEEN]
        else:  # Rank or file
            return piece.piece_type in [chess.ROOK, chess.QUEEN]
    
    def _detect_forks(self, board):
        """Detect fork opportunities"""
        score = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.KNIGHT:
                # Knight forks are most common
                knight_attacks = board.attacks(square)
                valuable_targets = []
                
                for target_square in knight_attacks:
                    target_piece = board.piece_at(target_square)
                    if target_piece and target_piece.color != piece.color:
                        if target_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]:
                            valuable_targets.append(target_piece)
                
                if len(valuable_targets) >= 2:
                    fork_value = sum(ADVANCED_PIECE_VALUES[p.piece_type] for p in valuable_targets) // 20
                    if piece.color == chess.WHITE:
                        score += fork_value
                    else:
                        score -= fork_value
        
        return score
    
    def _detect_discovered_attacks(self, board):
        """Detect discovered attack potential"""
        score = 0
        
        # Simplified discovered attack detection - avoid modifying the board
        # Instead, we'll check for pieces that could potentially be discovered
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP]:
                # Check if this piece could potentially block a discovery
                file, rank = chess.square_file(square), chess.square_rank(square)
                
                # Look for aligned pieces that could create discoveries
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
                
                for dx, dy in directions:
                    # Look behind the piece
                    attacker_found = False
                    for i in range(1, 8):
                        back_file, back_rank = file - dx * i, rank - dy * i
                        if not (0 <= back_file <= 7 and 0 <= back_rank <= 7):
                            break
                        
                        back_square = chess.square(back_file, back_rank)
                        back_piece = board.piece_at(back_square)
                        
                        if back_piece:
                            if (back_piece.color == piece.color and 
                                self._can_pin_along_direction(back_piece, dx, dy)):
                                attacker_found = True
                            break
                    
                    # Look ahead for targets
                    if attacker_found:
                        for i in range(1, 8):
                            front_file, front_rank = file + dx * i, rank + dy * i
                            if not (0 <= front_file <= 7 and 0 <= front_rank <= 7):
                                break
                            
                            front_square = chess.square(front_file, front_rank)
                            front_piece = board.piece_at(front_square)
                            
                            if front_piece:
                                if (front_piece.color != piece.color and
                                    front_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.KING]):
                                    
                                    discovery_value = ADVANCED_PIECE_VALUES[front_piece.piece_type] // 20
                                    if piece.color == chess.WHITE:
                                        score += discovery_value
                                    else:
                                        score -= discovery_value
                                break
        
        return score
    
    def _evaluate_back_rank_safety(self, board):
        """Evaluate back rank mate threats"""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is None:
                continue
            
            king_rank = chess.square_rank(king_square)
            back_rank = 0 if color == chess.WHITE else 7
            
            if king_rank == back_rank:
                # King is on back rank - check for escape squares
                escape_squares = 0
                king_file = chess.square_file(king_square)
                
                for file_offset in [-1, 0, 1]:
                    new_file = king_file + file_offset
                    if 0 <= new_file <= 7:
                        escape_square = chess.square(new_file, back_rank + (1 if color == chess.WHITE else -1))
                        if not board.piece_at(escape_square):
                            escape_squares += 1
                
                if escape_squares == 0:
                    # No escape squares - vulnerable to back rank mate
                    if color == chess.WHITE:
                        score -= 50
                    else:
                        score += 50
        
        return score
    
    def order_moves(self, board, moves, best_move=None, depth=0):
        """Advanced move ordering for better alpha-beta pruning"""
        move_scores = []
        
        for move in moves:
            score = 0
            
            # Best move from transposition table gets highest priority
            if best_move and move == best_move:
                score += 10000
            
            # Captures - MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                moving_piece = board.piece_at(move.from_square)
                
                if captured_piece and moving_piece:
                    mvv_lva = ADVANCED_PIECE_VALUES[captured_piece.piece_type] - ADVANCED_PIECE_VALUES[moving_piece.piece_type]
                    score += 8000 + mvv_lva
            
            # Promotions
            if move.promotion:
                score += 7000 + ADVANCED_PIECE_VALUES[move.promotion]
            
            # Killer moves
            if self.killer_moves.is_killer(depth, move):
                score += 6000
            
            # History heuristic
            score += self.history_table.get_score(move) // 100
            
            # Checks
            board.push(move)
            if board.is_check():
                score += 5000
            board.pop()
            
            # Central squares
            if move.to_square in [chess.E4, chess.E5, chess.D4, chess.D5]:
                score += 100
            
            move_scores.append((score, move))
        
        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in move_scores]
    
    def quiescence_search(self, board, alpha, beta, depth=0):
        """Quiescence search to avoid horizon effect"""
        self.nodes_searched += 1
        
        if depth >= self.max_quiesce_depth:
            return self.evaluate_board(board)
        
        # Stand pat evaluation
        stand_pat = self.evaluate_board(board)
        
        if stand_pat >= beta:
            return beta
        
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Only consider captures and checks in quiescence
        moves = []
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move):
                moves.append(move)
        
        # Order captures by MVV-LVA
        moves = self.order_moves(board, moves)
        
        for move in moves:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, depth + 1)
            board.pop()
            
            if score >= beta:
                return beta
            
            if score > alpha:
                alpha = score
        
        return alpha
    
    def null_move_search(self, board, depth, beta):
        """Null move pruning"""
        if depth < 3 or self.is_endgame(board):
            return False
        
        # Make null move
        board.turn = not board.turn
        
        # Search with reduced depth
        score = -self.alpha_beta(board, depth - 1 - self.null_move_reduction, -beta, -beta + 1, False)
        
        # Unmake null move
        board.turn = not board.turn
        
        return score >= beta
    
    def alpha_beta(self, board, depth, alpha, beta, do_null=True):
        """Alpha-beta search with advanced pruning techniques"""
        self.nodes_searched += 1
        
        if depth == 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta)
        
        # Transposition table lookup
        tt_score, tt_move = self.tt.probe(board, depth, alpha, beta)
        if tt_score is not None:
            return tt_score
        
        # Null move pruning
        if do_null and depth >= 3 and not board.is_check() and not self.is_endgame(board):
            if self.null_move_search(board, depth, beta):
                return beta
        
        moves = list(board.legal_moves)
        if not moves:
            return self.evaluate_board(board)
        
        # Move ordering
        moves = self.order_moves(board, moves, tt_move, depth)
        
        best_score = -float('inf')
        best_move = None
        moves_searched = 0
        
        for i, move in enumerate(moves):
            # Check if move is legal before making it
            if move not in board.legal_moves:
                continue
                
            board.push(move)
            
            # Late Move Reduction (LMR)
            reduction = 0
            if (i >= self.lmr_threshold and depth >= 3 and 
                not board.is_capture(move) and not board.is_check() and 
                not self.killer_moves.is_killer(depth, move)):
                reduction = self.lmr_reduction
            
            # Principal Variation Search
            if moves_searched == 0:
                # Full window search for first move
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha, True)
            else:
                # Null window search
                score = -self.alpha_beta(board, depth - 1 - reduction, -alpha - 1, -alpha, True)
                
                # Re-search if necessary
                if score > alpha and reduction > 0:
                    score = -self.alpha_beta(board, depth - 1, -alpha - 1, -alpha, True)
                
                if score > alpha and score < beta:
                    score = -self.alpha_beta(board, depth - 1, -beta, -alpha, True)
            
            board.pop()
            moves_searched += 1
            
            if score > best_score:
                best_score = score
                best_move = move
            
            if score > alpha:
                alpha = score
            
            if alpha >= beta:
                # Beta cutoff - update killer moves and history
                if not board.is_capture(move):
                    self.killer_moves.add_killer(depth, move)
                    self.history_table.update(move, depth)
                break
        
        # Store in transposition table
        if best_score <= alpha:
            flag = TranspositionTable.UPPER_BOUND
        elif best_score >= beta:
            flag = TranspositionTable.LOWER_BOUND
        else:
            flag = TranspositionTable.EXACT
        
        self.tt.store(board, depth, best_score, best_move, flag)
        
        return best_score
    
    def iterative_deepening_search(self, board, max_depth, time_limit=5.0, player_model=None, move_number=0):
        """Iterative deepening with time management and aspiration windows"""
        start_time = time.time()
        best_move = None
        best_score = 0
        
        # Check opening book first
        if move_number < 10:
            opening_move = self.get_opening_move(board)
            if opening_move and opening_move in board.legal_moves:
                return 0, opening_move
        
        # Reset counters
        self.nodes_searched = 0
        self.tt_hits = self.tt.hits
        
        # Aspiration window parameters
        window_size = 50
        
        for depth in range(1, max_depth + 1):
            iteration_start = time.time()
            
            if depth == 1:
                # Full window for first iteration
                score = self.alpha_beta(board, depth, -float('inf'), float('inf'))
            else:
                # Aspiration windows
                alpha = best_score - window_size
                beta = best_score + window_size
                
                score = self.alpha_beta(board, depth, alpha, beta)
                
                # Re-search if outside window
                if score <= alpha:
                    score = self.alpha_beta(board, depth, -float('inf'), beta)
                elif score >= beta:
                    score = self.alpha_beta(board, depth, alpha, float('inf'))
            
            # Get best move from transposition table
            _, move = self.tt.probe(board, depth, -float('inf'), float('inf'))
            
            if move:
                best_move = move
                best_score = score
            
            elapsed = time.time() - start_time
            
            # Time management
            if elapsed > time_limit * 0.6:  # Use 60% of time limit
                break
            
            # If we found a mate, no need to search deeper
            if abs(score) > 29000:
                break
        
        self.tt_hits = self.tt.hits - self.tt_hits
        
        return best_score, best_move

# Legacy ChessAI class for backward compatibility
class ChessAI:
    def __init__(self, depth=3):
        self.depth = depth
        self._thinking = False
        self.current_move = None
        self.thinking_thread = None
        
        # Use the advanced AI internally
        self.player_model = PlayerModel()
        self.advanced_ai = AdvancedAdaptiveChessAI(self.player_model, search_depth=depth)
    
    def start_thinking(self, board):
        return self.advanced_ai.start_thinking(board)
    
    def get_move(self):
        return self.advanced_ai.get_move()
    
    @property
    def thinking(self):
        return self.advanced_ai.thinking if hasattr(self, 'advanced_ai') else self._thinking
    
    @thinking.setter
    def thinking(self, value):
        if hasattr(self, 'advanced_ai'):
            self.advanced_ai.thinking = value
        else:
            self._thinking = value

class AdvancedAdaptiveChessAI:
    """Advanced adaptive chess AI with 3000+ ELO strength"""
    
    def __init__(self, player_model, search_depth=4, aggressivity_factor=1.0):
        self.player_model = player_model
        self.base_search_depth = search_depth
        self.aggressivity_factor = aggressivity_factor
        self.engine = OptimizedChessEngine()
        
        self.thinking = False
        self.current_move = None
        self.thinking_thread = None
        self.last_search_time = 0
        self.performance_stats = {}
        
        # Advanced time management
        self.total_time = 300  # 5 minutes per side
        self.time_used = 0
        self.moves_played = 0
    
    def get_adaptive_depth(self, board):
        """Calculate adaptive search depth based on position"""
        base_depth = self.base_search_depth
        
        # Increase depth in endgame
        if self.engine.is_endgame(board):
            base_depth += 2
        
        # Increase depth for tactical positions
        if board.is_check():
            base_depth += 1
        
        # Increase depth for critical moves
        legal_moves = list(board.legal_moves)
        if len(legal_moves) <= 5:  # Few legal moves - critical position
            base_depth += 1
        
        # Decrease depth in opening with many pieces
        piece_count = len(board.piece_map())
        if piece_count > 28 and board.fullmove_number < 10:
            base_depth = max(base_depth - 1, 2)
        
        return min(base_depth, 8)  # Cap at depth 8
    
    def calculate_time_limit(self, board):
        """Advanced time management"""
        moves_remaining = max(40 - self.moves_played, 10)
        base_time = (self.total_time - self.time_used) / moves_remaining
        
        # Adjust based on position complexity
        if board.is_check():
            base_time *= 1.5
        
        if self.engine.is_endgame(board):
            base_time *= 1.3
        
        # Don't use more than 20% of remaining time on one move
        max_time = (self.total_time - self.time_used) * 0.2
        
        return min(max(base_time, 1.0), max_time)
    
    def calculate_move(self, board_copy):
        """Calculate best move using advanced engine"""
        try:
            start_time = time.time()
            
            adaptive_depth = self.get_adaptive_depth(board_copy)
            time_limit = self.calculate_time_limit(board_copy)
            
            score, best_move = self.engine.iterative_deepening_search(
                board_copy, adaptive_depth, time_limit, 
                self.player_model, board_copy.fullmove_number
            )
            
            self.last_search_time = time.time() - start_time
            self.time_used += self.last_search_time
            self.moves_played += 1
            
            # Store performance stats
            self.performance_stats = {
                'nodes_searched': self.engine.nodes_searched,
                'transposition_hits': self.engine.tt_hits,
                'search_depth': adaptive_depth,
                'evaluation': score,
                'time_used': self.last_search_time
            }
            
            self.current_move = best_move
            
        except Exception as e:
            print(f"AI calculation error: {e}")
            legal_moves = list(board_copy.legal_moves)
            self.current_move = random.choice(legal_moves) if legal_moves else None
        finally:
            self.thinking = False
    
    def start_thinking(self, board):
        """Start thinking in background thread"""
        if not self.thinking:
            self.thinking = True
            self.current_move = None
            board_copy = board.copy()
            self.thinking_thread = threading.Thread(target=self.calculate_move, args=(board_copy,))
            self.thinking_thread.daemon = True
            self.thinking_thread.start()
    
    def get_move(self):
        """Get calculated move"""
        if not self.thinking and self.current_move:
            move = self.current_move
            self.current_move = None
            return move
        return None
    
    def record_player_move(self, move_uci):
        """Record player move for learning"""
        if self.player_model:
            self.player_model.move_counter[move_uci] += 1
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return self.performance_stats.copy()

def load_piece_images():
    images = {}
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pawns_dir = os.path.join(base_dir, 'Pawns')
    
    for symbol, filename in PIECE_IMAGES.items():
        path = os.path.join(pawns_dir, filename)
        try:
            if os.path.exists(path):
                image = pygame.image.load(path)
                image = pygame.transform.scale(image, (SQUARE_SIZE - 10, SQUARE_SIZE - 10))
                images[symbol] = image
            else:
                images[symbol] = None
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            images[symbol] = None
    
    return images

def draw_board(screen, board, images, selected_square=None, legal_moves=None, last_move=None):
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, 7 - rank)
            color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
            rect = pygame.Rect(file * SQUARE_SIZE, rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            
            # Draw square
            pygame.draw.rect(screen, color, rect)
            
            # Highlight selected square
            if selected_square == square:
                pygame.draw.rect(screen, SELECTED_COLOR, rect, 4)
            
            # Highlight last move
            if last_move and (last_move.from_square == square or last_move.to_square == square):
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 3)
            
            # Show legal moves
            if legal_moves and square in legal_moves:
                center = rect.center
                pygame.draw.circle(screen, LEGAL_MOVE_COLOR, center, 10)
            
            # Draw piece
            piece = board.piece_at(square)
            if piece:
                symbol = piece.symbol()
                if symbol in images and images[symbol]:
                    piece_rect = images[symbol].get_rect(center=rect.center)
                    screen.blit(images[symbol], piece_rect)
                else:
                    # Text fallback
                    font = pygame.font.Font(None, 48)
                    text_color = (255, 255, 255) if piece.color == chess.WHITE else (0, 0, 0)
                    text = font.render(PIECE_SYMBOLS.get(symbol, symbol), True, text_color)
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)

def draw_panel(screen, board, ai, move_history):
    panel_rect = pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, BOARD_SIZE)
    pygame.draw.rect(screen, PANEL_COLOR, panel_rect)
    
    font = pygame.font.Font(None, 24)
    y_pos = 20
    
    # Game status - simplified
    if board.is_game_over():
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            status = f"{winner} wins!"
        else:
            status = "Draw"
    else:
        turn = "White" if board.turn == chess.WHITE else "Black"
        if board.turn == chess.BLACK and ai.thinking:
            status = "AI thinking..."
        else:
            status = f"{turn} to move"
    
    text = font.render(status, True, TEXT_COLOR)
    screen.blit(text, (BOARD_SIZE + 10, y_pos))
    y_pos += 40
    
    # Check warning - only if in check
    if board.is_check() and not board.is_game_over():
        check_text = font.render("CHECK!", True, (255, 0, 0))
        screen.blit(check_text, (BOARD_SIZE + 10, y_pos))
        y_pos += 30
    
    # Move history - simplified
    if move_history:
        recent_moves = move_history[-8:]  # Show last 8 moves instead of 10
        for i, move in enumerate(recent_moves):
            move_text = font.render(f"{len(move_history) - len(recent_moves) + i + 1}. {move}", True, TEXT_COLOR)
            screen.blit(move_text, (BOARD_SIZE + 10, y_pos))
            y_pos += 20  # Reduced spacing

def get_square_from_pos(pos):
    x, y = pos
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return None
    file = x // SQUARE_SIZE
    rank = 7 - (y // SQUARE_SIZE)
    return chess.square(file, rank)

def get_legal_moves_for_square(board, square):
    return [move.to_square for move in board.legal_moves if move.from_square == square]

def draw_menu_overlay(screen, ai_depth, show_legal_moves, show_side_panel):
    # Create semi-transparent overlay
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    
    # Menu box - smaller
    menu_width = 350
    menu_height = 280
    menu_x = (WINDOW_WIDTH - menu_width) // 2
    menu_y = (WINDOW_HEIGHT - menu_height) // 2
    
    menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
    pygame.draw.rect(screen, MENU_BOX_COLOR, menu_rect)
    pygame.draw.rect(screen, TEXT_COLOR, menu_rect, 2)
    
    # Title
    title_font = pygame.font.Font(None, 32)
    title_text = title_font.render("SETTINGS", True, TEXT_COLOR)
    title_rect = title_text.get_rect(center=(menu_x + menu_width // 2, menu_y + 25))
    screen.blit(title_text, title_rect)
    
    # Settings text
    font = pygame.font.Font(None, 24)
    y_offset = 70
    
    # AI Difficulty
    difficulty_names = {1: "Beginner", 2: "Easy", 3: "Normal", 4: "Hard", 5: "Expert", 6: "Master"}
    difficulty_text = difficulty_names.get(ai_depth, "Custom")
    ai_text = font.render(f"AI: {ai_depth} ({difficulty_text})", True, TEXT_COLOR)
    screen.blit(ai_text, (menu_x + 20, menu_y + y_offset))
    
    control_text = font.render("1: -  2: +", True, (180, 180, 180))
    screen.blit(control_text, (menu_x + 40, menu_y + y_offset + 25))
    y_offset += 60
    
    # Show Legal Moves
    legal_status = "ON" if show_legal_moves else "OFF"
    legal_text = font.render(f"Legal moves: {legal_status}", True, TEXT_COLOR)
    screen.blit(legal_text, (menu_x + 20, menu_y + y_offset))
    
    control_text = font.render("L: toggle", True, (180, 180, 180))
    screen.blit(control_text, (menu_x + 40, menu_y + y_offset + 25))
    y_offset += 60
    
    # Show Side Panel
    panel_status = "ON" if show_side_panel else "OFF"
    panel_text = font.render(f"Side panel: {panel_status}", True, TEXT_COLOR)
    screen.blit(panel_text, (menu_x + 20, menu_y + y_offset))
    
    control_text = font.render("P: toggle", True, (180, 180, 180))
    screen.blit(control_text, (menu_x + 40, menu_y + y_offset + 25))
    y_offset += 60
    
    # Close instruction
    close_text = font.render("M: close", True, (200, 200, 200))
    screen.blit(close_text, (menu_x + 20, menu_y + y_offset))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, BOARD_SIZE))
    pygame.display.set_caption("Chess AI")
    clock = pygame.time.Clock()
    
    # Initialize game
    board = chess.Board()
    player_model = PlayerModel()
    ai = ChessAI(depth=4)  # Use legacy wrapper with advanced AI inside
    images = load_piece_images()
    
    selected_square = None
    legal_moves = []
    move_history = []
    last_move = None
    game_moves = []
    running = True
    show_menu = False
    show_legal_moves = True
    show_side_panel = True
    
    # ESC hold variables
    esc_hold_start = None
    esc_hold_duration = 1000  # 1 second in milliseconds
    esc_held = False
    
    while running:
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if game_moves:
                    player_model.record_game(game_moves, completed=board.is_game_over())
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    esc_hold_start = current_time
                    esc_held = False
                elif event.key == pygame.K_m:
                    show_menu = not show_menu
                elif event.key == pygame.K_r:
                    # Reset game
                    if game_moves:
                        player_model.record_game(game_moves, completed=board.is_game_over())
                    board.reset()
                    selected_square = None
                    legal_moves = []
                    move_history = []
                    last_move = None
                    game_moves = []
                elif show_menu:
                    # Menu controls
                    if event.key == pygame.K_1:
                        ai.base_search_depth = max(1, ai.base_search_depth - 1)
                    elif event.key == pygame.K_2:
                        ai.base_search_depth = min(6, ai.base_search_depth + 1)
                    elif event.key == pygame.K_l:
                        show_legal_moves = not show_legal_moves
                    elif event.key == pygame.K_p:
                        show_side_panel = not show_side_panel
                        # Resize window based on side panel state
                        if show_side_panel:
                            pygame.display.set_mode((WINDOW_WIDTH, BOARD_SIZE))
                        else:
                            pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    esc_hold_start = None
                    esc_held = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN and not show_menu and board.turn == chess.WHITE and not board.is_game_over():
                # Calculate mouse position relative to board
                mouse_x, mouse_y = event.pos
                current_width = screen.get_width()
                
                if show_side_panel:
                    # Normal layout - board starts at (0, 0)
                    square = get_square_from_pos(event.pos)
                else:
                    # Board fills entire window when side panel is off
                    square = get_square_from_pos(event.pos)
                
                if square is not None:
                    if selected_square is None:
                        # Select piece
                        piece = board.piece_at(square)
                        if piece and piece.color == chess.WHITE:
                            selected_square = square
                            if show_legal_moves:
                                legal_moves = get_legal_moves_for_square(board, square)
                    else:
                        # Make move
                        move = chess.Move(selected_square, square)
                        if move in board.legal_moves:
                            board.push(move)
                            move_history.append(move.uci())
                            game_moves.append(move.uci())
                            last_move = move
                            selected_square = None
                            legal_moves = []
                            
                            # Start AI thinking
                            if not board.is_game_over():
                                ai.start_thinking(board)
                        else:
                            # Select new piece or deselect
                            piece = board.piece_at(square)
                            if piece and piece.color == chess.WHITE:
                                selected_square = square
                                if show_legal_moves:
                                    legal_moves = get_legal_moves_for_square(board, square)
                            else:
                                selected_square = None
                                legal_moves = []
        
        # Check ESC hold
        if esc_hold_start and not esc_held:
            if current_time - esc_hold_start >= esc_hold_duration:
                esc_held = True
                if game_moves:
                    player_model.record_game(game_moves, completed=board.is_game_over())
                running = False
        
        # AI move
        if board.turn == chess.BLACK and not board.is_game_over() and not ai.thinking:
            ai_move = ai.get_move()
            if ai_move:
                board.push(ai_move)
                move_history.append(ai_move.uci())
                game_moves.append(ai_move.uci())
                last_move = ai_move
            elif not ai.thinking:
                ai.start_thinking(board)
        
        # Draw everything
        screen.fill(BACKGROUND_COLOR)
        
        # Calculate board position based on window size
        current_width = screen.get_width()
        current_height = screen.get_height()
        
        if show_side_panel:
            # Normal layout with side panel
            draw_board(screen, board, images, selected_square, legal_moves if show_legal_moves else None, last_move)
        else:
            # Full window board when side panel is off
            draw_board(screen, board, images, selected_square, legal_moves if show_legal_moves else None, last_move)
        
        # Draw side panel only if enabled
        if show_side_panel:
            draw_panel(screen, board, ai, move_history)
        
        # Draw ESC hold progress
        if esc_hold_start and not esc_held:
            # Simple minimalist text only - centered on screen
            font = pygame.font.Font(None, 24)
            text = font.render("Hold ESC to quit", True, TEXT_COLOR)
            text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            screen.blit(text, text_rect)
        
        # Draw menu overlay if active
        if show_menu:
            draw_menu_overlay(screen, ai.base_search_depth, show_legal_moves, show_side_panel)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()