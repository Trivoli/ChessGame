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

# File paths
PLAYER_MODEL_FILE = 'player_model.json'
PLAYER_GAMES_LOG = 'player_games.log'

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

class ChessAI:
    def __init__(self, depth=3):
        self.depth = depth
        self.thinking = False
        self.current_move = None
        self.thinking_thread = None

    def evaluate_board(self, board):
        if board.is_checkmate():
            return -30000 if board.turn else 30000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = PIECE_VALUES[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value

        # Add mobility bonus
        white_moves = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
        board.turn = not board.turn
        black_moves = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
        board.turn = not board.turn
        
        score += (white_moves - black_moves) * 10
        return score

    def minimax(self, board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board), None
        
        best_move = None
        if maximizing:
            max_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                    
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval_score, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def calculate_move(self, board_copy):
        try:
            _, best_move = self.minimax(board_copy, self.depth, -float('inf'), float('inf'), 
                                      board_copy.turn == chess.WHITE)
            self.current_move = best_move
        except Exception as e:
            print(f"AI error: {e}")
            legal_moves = list(board_copy.legal_moves)
            self.current_move = random.choice(legal_moves) if legal_moves else None
        finally:
            self.thinking = False

    def start_thinking(self, board):
        if not self.thinking:
            self.thinking = True
            self.current_move = None
            board_copy = board.copy()
            self.thinking_thread = threading.Thread(target=self.calculate_move, args=(board_copy,))
            self.thinking_thread.daemon = True
            self.thinking_thread.start()

    def get_move(self):
        if not self.thinking and self.current_move:
            move = self.current_move
            self.current_move = None
            return move
        return None

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
    ai = ChessAI(depth=3)
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
                        ai.depth = max(1, ai.depth - 1)
                    elif event.key == pygame.K_2:
                        ai.depth = min(6, ai.depth + 1)
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
            draw_menu_overlay(screen, ai.depth, show_legal_moves, show_side_panel)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()