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

# Minimalistic color palette - Modern & Clean
LIGHT_SQUARE = (245, 245, 245)      # Clean white
DARK_SQUARE = (120, 120, 120)       # Elegant gray
HIGHLIGHT_COLOR = (255, 206, 84)    # Warm gold
SELECTED_COLOR = (74, 144, 226)     # Soft blue
LEGAL_MOVE_COLOR = (74, 144, 226)   # Consistent blue
BACKGROUND_COLOR = (248, 248, 248)  # Off-white background
TEXT_COLOR = (60, 60, 60)           # Dark gray text
TEXT_SECONDARY = (140, 140, 140)    # Light gray for secondary text
PANEL_COLOR = (255, 255, 255)       # Pure white panel
PANEL_BORDER = (220, 220, 220)      # Subtle border
MENU_OVERLAY_COLOR = (0, 0, 0, 120) # Subtle overlay
MENU_BOX_COLOR = (255, 255, 255)    # Clean white menu
ACCENT_COLOR = (74, 144, 226)       # Primary accent
SUCCESS_COLOR = (46, 160, 67)       # Green for success
WARNING_COLOR = (255, 149, 0)       # Orange for warnings
ERROR_COLOR = (255, 59, 48)         # Red for errors

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

def draw_board(screen, board, images, selected_square=None, legal_moves=None, last_move=None, 
               hovered_square=None, dragging_piece=None, drag_offset=(0, 0), board_flipped=False):
    # Draw board with subtle shadow effect
    shadow_offset = 4
    shadow_rect = pygame.Rect(shadow_offset, shadow_offset, BOARD_SIZE, BOARD_SIZE)
    pygame.draw.rect(screen, (200, 200, 200, 50), shadow_rect)
    
    for rank in range(8):
        for file in range(8):
            # Calculate square based on board orientation
            if board_flipped:
                square = chess.square(7 - file, rank)
                display_rank, display_file = rank, 7 - file
            else:
                square = chess.square(file, 7 - rank)
                display_rank, display_file = rank, file
                
            color = LIGHT_SQUARE if (display_rank + display_file) % 2 == 0 else DARK_SQUARE
            rect = pygame.Rect(display_file * SQUARE_SIZE, display_rank * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            
            # Draw square with subtle gradient effect
            pygame.draw.rect(screen, color, rect)
            
            # Add subtle inner shadow for depth
            if (rank + file) % 2 == 1:  # Dark squares only
                inner_rect = pygame.Rect(rect.x + 1, rect.y + 1, rect.width - 2, rect.height - 2)
                pygame.draw.rect(screen, (110, 110, 110), inner_rect)
            
            # Subtle hover effect
            if hovered_square == square and not selected_square:
                hover_surface = pygame.Surface((rect.width, rect.height))
                hover_surface.set_alpha(20)
                hover_surface.fill((74, 144, 226))  # Subtle blue tint
                screen.blit(hover_surface, rect)
            
            # Highlight selected square with elegant glow
            if selected_square == square:
                # Outer glow
                glow_rect = pygame.Rect(rect.x - 2, rect.y - 2, rect.width + 4, rect.height + 4)
                pygame.draw.rect(screen, (*SELECTED_COLOR, 100), glow_rect)
                # Inner highlight
                pygame.draw.rect(screen, SELECTED_COLOR, rect, 3)
            
            # Highlight last move with subtle golden accent
            if last_move and (last_move.from_square == square or last_move.to_square == square):
                highlight_surface = pygame.Surface((rect.width, rect.height))
                highlight_surface.set_alpha(80)
                highlight_surface.fill(HIGHLIGHT_COLOR)
                screen.blit(highlight_surface, rect)
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 2)
            
            # Show legal moves with elegant dots
            if legal_moves and square in legal_moves:
                center = rect.center
                # Check if there's a piece on the target square (capture move)
                target_piece = board.piece_at(square)
                if target_piece:
                    # Capture indicator - ring around the square
                    pygame.draw.circle(screen, LEGAL_MOVE_COLOR, center, SQUARE_SIZE // 3, 4)
                else:
                    # Normal move indicator - small filled circle
                    pygame.draw.circle(screen, LEGAL_MOVE_COLOR, center, 8)
                    # Add subtle outer ring
                    pygame.draw.circle(screen, (*LEGAL_MOVE_COLOR, 100), center, 12, 2)
            
            # Draw piece (skip if it's being dragged)
            piece = board.piece_at(square)
            if piece and not (dragging_piece and square == selected_square):
                symbol = piece.symbol()
                if symbol in images and images[symbol]:
                    # Add subtle drop shadow to pieces
                    shadow_rect = images[symbol].get_rect(center=(rect.centerx + 1, rect.centery + 1))
                    shadow_surface = images[symbol].copy()
                    shadow_surface.fill((0, 0, 0, 30), special_flags=pygame.BLEND_RGBA_MULT)
                    screen.blit(shadow_surface, shadow_rect)
                    
                    # Draw the actual piece
                    piece_rect = images[symbol].get_rect(center=rect.center)
                    screen.blit(images[symbol], piece_rect)
                else:
                    # Enhanced text fallback with better typography
                    font = pygame.font.Font(None, 52)
                    text_color = (40, 40, 40) if piece.color == chess.WHITE else (240, 240, 240)
                    text = font.render(PIECE_SYMBOLS.get(symbol, symbol), True, text_color)
                    
                    # Add text shadow
                    shadow_text = font.render(PIECE_SYMBOLS.get(symbol, symbol), True, (0, 0, 0, 50))
                    shadow_rect = shadow_text.get_rect(center=(rect.centerx + 1, rect.centery + 1))
                    screen.blit(shadow_text, shadow_rect)
                    
                    # Draw main text
                    text_rect = text.get_rect(center=rect.center)
                    screen.blit(text, text_rect)
    
    # Draw dragged piece at mouse position
    if dragging_piece and selected_square is not None:
        if board_flipped:
            drag_x = (7 - selected_square % 8) * SQUARE_SIZE + SQUARE_SIZE // 2 + drag_offset[0]
            drag_y = (selected_square // 8) * SQUARE_SIZE + SQUARE_SIZE // 2 + drag_offset[1]
        else:
            drag_x = (selected_square % 8) * SQUARE_SIZE + SQUARE_SIZE // 2 + drag_offset[0]
            drag_y = (7 - selected_square // 8) * SQUARE_SIZE + SQUARE_SIZE // 2 + drag_offset[1]
        
        symbol = dragging_piece.symbol()
        if symbol in images and images[symbol]:
            # Enhanced shadow for dragged piece
            shadow_rect = images[symbol].get_rect(center=(drag_x + 2, drag_y + 2))
            shadow_surface = images[symbol].copy()
            shadow_surface.set_alpha(100)
            shadow_surface.fill((0, 0, 0), special_flags=pygame.BLEND_RGBA_MULT)
            screen.blit(shadow_surface, shadow_rect)
            
            # Draw dragged piece with slight transparency
            piece_surface = images[symbol].copy()
            piece_surface.set_alpha(220)  # Slightly transparent
            piece_rect = piece_surface.get_rect(center=(drag_x, drag_y))
            screen.blit(piece_surface, piece_rect)
        else:
            # Text fallback for dragged piece
            font = pygame.font.Font(None, 52)
            text_color = (40, 40, 40) if dragging_piece.color == chess.WHITE else (240, 240, 240)
            text = font.render(PIECE_SYMBOLS.get(symbol, symbol), True, text_color)
            text_rect = text.get_rect(center=(drag_x, drag_y))
            screen.blit(text, text_rect)
    
    # Draw board border with elegant styling
    border_color = PANEL_BORDER
    pygame.draw.rect(screen, border_color, (0, 0, BOARD_SIZE, BOARD_SIZE), 2)

def draw_coordinates(screen, show_coordinates, board_flipped=False):
    """Draw board coordinates (a-h, 1-8) around the board"""
    if not show_coordinates:
        return
        
    font = pygame.font.Font(None, 18)
    coord_color = TEXT_SECONDARY
    
    # File labels (a-h) at bottom
    for file in range(8):
        if board_flipped:
            label = chr(ord('h') - file)  # h-a when flipped
            x = file * SQUARE_SIZE + SQUARE_SIZE // 2
        else:
            label = chr(ord('a') + file)  # a-h normally
            x = file * SQUARE_SIZE + SQUARE_SIZE // 2
        y = BOARD_SIZE + 8
        text = font.render(label, True, coord_color)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)
    
    # Rank labels (1-8) on the left
    for rank in range(8):
        if board_flipped:
            label = str(rank + 1)  # 1-8 when flipped
        else:
            label = str(8 - rank)  # 8-1 normally
        x = -15
        y = rank * SQUARE_SIZE + SQUARE_SIZE // 2
        text = font.render(label, True, coord_color)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)

def draw_panel(screen, board, ai, move_history, captured_white=None, captured_black=None, 
               show_move_timer=True, last_move_time=0, current_time=0, show_captured_pieces=True, show_coordinates=False):
    panel_rect = pygame.Rect(BOARD_SIZE, 0, PANEL_WIDTH, BOARD_SIZE)
    
    # Draw panel background with subtle gradient
    pygame.draw.rect(screen, PANEL_COLOR, panel_rect)
    
    # Add subtle left border
    pygame.draw.line(screen, PANEL_BORDER, (BOARD_SIZE, 0), (BOARD_SIZE, BOARD_SIZE), 1)
    
    # Modern typography
    title_font = pygame.font.Font(None, 28)
    font = pygame.font.Font(None, 22)
    small_font = pygame.font.Font(None, 18)
    
    margin = 24
    y_pos = margin
    
    # Game status with modern styling
    if board.is_game_over():
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            status = f"{winner} wins!"
            status_color = SUCCESS_COLOR
        else:
            status = "Draw"
            status_color = TEXT_SECONDARY
    else:
        turn = "White" if board.turn == chess.WHITE else "Black"
        if board.turn == chess.BLACK and ai.thinking:
            status = "AI thinking..."
            status_color = ACCENT_COLOR
            # Add thinking animation dots
            dots = "." * ((pygame.time.get_ticks() // 500) % 4)
            status += dots
        else:
            status = f"{turn} to move"
            status_color = TEXT_COLOR
    
    # Status card background
    status_rect = pygame.Rect(BOARD_SIZE + margin//2, y_pos - 8, PANEL_WIDTH - margin, 40)
    pygame.draw.rect(screen, (250, 250, 250), status_rect, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER, status_rect, 1, border_radius=8)
    
    text = title_font.render(status, True, status_color)
    text_rect = text.get_rect(center=(BOARD_SIZE + PANEL_WIDTH//2, y_pos + 12))
    screen.blit(text, text_rect)
    y_pos += 60
    
    # Check warning with modern alert styling
    if board.is_check() and not board.is_game_over():
        check_rect = pygame.Rect(BOARD_SIZE + margin//2, y_pos - 8, PANEL_WIDTH - margin, 32)
        pygame.draw.rect(screen, (255, 245, 245), check_rect, border_radius=6)
        pygame.draw.rect(screen, ERROR_COLOR, check_rect, 2, border_radius=6)
        
        check_text = font.render("⚠ CHECK!", True, ERROR_COLOR)
        check_text_rect = check_text.get_rect(center=(BOARD_SIZE + PANEL_WIDTH//2, y_pos + 8))
        screen.blit(check_text, check_text_rect)
        y_pos += 50
    
    # Move history section with clean styling
    if move_history:
        # Section header
        history_title = font.render("Move History", True, TEXT_SECONDARY)
        screen.blit(history_title, (BOARD_SIZE + margin, y_pos))
        y_pos += 35
        
        # History container
        history_height = min(len(move_history) * 24 + 20, 300)
        history_rect = pygame.Rect(BOARD_SIZE + margin//2, y_pos - 10, PANEL_WIDTH - margin, history_height)
        pygame.draw.rect(screen, (252, 252, 252), history_rect, border_radius=8)
        pygame.draw.rect(screen, PANEL_BORDER, history_rect, 1, border_radius=8)
        
        # Show recent moves with alternating background
        recent_moves = move_history[-12:]  # Show last 12 moves
        for i, move in enumerate(recent_moves):
            move_y = y_pos + (i * 24)
            if move_y > BOARD_SIZE - 40:  # Don't overflow panel
                break
                
            # Alternating row background
            if i % 2 == 0:
                row_rect = pygame.Rect(BOARD_SIZE + margin//2 + 4, move_y - 2, PANEL_WIDTH - margin - 8, 20)
                pygame.draw.rect(screen, (248, 248, 248), row_rect, border_radius=4)
            
            move_number = len(move_history) - len(recent_moves) + i + 1
            move_text = small_font.render(f"{move_number:2d}. {move}", True, TEXT_COLOR)
            screen.blit(move_text, (BOARD_SIZE + margin + 8, move_y))
        
        # Subtle scroll indicator if there are more moves
        if len(move_history) > 12:
            scroll_text = small_font.render(f"... and {len(move_history) - 12} more", True, TEXT_SECONDARY)
            screen.blit(scroll_text, (BOARD_SIZE + margin + 8, y_pos + len(recent_moves) * 24 + 5))
    
    # Show captured pieces if enabled
    if show_captured_pieces and (captured_white or captured_black):
        # Captured pieces section
        if y_pos < BOARD_SIZE - 200:  # Only show if there's space
            capture_title = font.render("Captured", True, TEXT_SECONDARY)
            screen.blit(capture_title, (BOARD_SIZE + margin, y_pos))
            y_pos += 30
            
            # Captured by player (black pieces)
            if captured_black:
                captured_text = small_font.render(f"You: {''.join(captured_black)}", True, TEXT_COLOR)
                screen.blit(captured_text, (BOARD_SIZE + margin + 8, y_pos))
                y_pos += 20
            
            # Captured by AI (white pieces)
            if captured_white:
                captured_text = small_font.render(f"AI: {''.join(captured_white)}", True, TEXT_COLOR)
                screen.blit(captured_text, (BOARD_SIZE + margin + 8, y_pos))
                y_pos += 20
    
    # Show move timer if enabled
    if show_move_timer and last_move_time > 0:
        time_since_move = (current_time - last_move_time) / 1000.0  # Convert to seconds
        if time_since_move < 60:  # Only show for recent moves
            timer_text = small_font.render(f"Last move: {time_since_move:.1f}s ago", True, TEXT_SECONDARY)
            footer_y = BOARD_SIZE - 80
            screen.blit(timer_text, (BOARD_SIZE + margin, footer_y))
    
    # Add subtle footer with controls hint - minimalistic
    footer_y = BOARD_SIZE - 60
    if show_move_timer or show_captured_pieces or show_coordinates:
        # Show active features only
        active_features = []
        if show_coordinates: active_features.append("Coords")
        if show_move_timer: active_features.append("Timer")
        if show_captured_pieces: active_features.append("Captures")
        footer_text = small_font.render(" | ".join(active_features), True, TEXT_SECONDARY)
    else:
        # Minimal hint when nothing is active
        footer_text = small_font.render("Press M for options", True, TEXT_SECONDARY)
    
    footer_rect = footer_text.get_rect(center=(BOARD_SIZE + PANEL_WIDTH//2, footer_y))
    screen.blit(footer_text, footer_rect)



def get_legal_moves_for_square(board, square):
    return [move.to_square for move in board.legal_moves if move.from_square == square]

def get_square_from_pos(pos, board_flipped=False):
    """Convert mouse position to chess square, accounting for board flipping"""
    x, y = pos
    if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
        return None
    
    file = x // SQUARE_SIZE
    rank = y // SQUARE_SIZE
    
    if board_flipped:
        return chess.square(7 - file, rank)
    else:
        return chess.square(file, 7 - rank)

def draw_menu_overlay(screen, ai_depth, show_legal_moves, show_side_panel, player_is_white=True, board_flipped=False):
    # Create elegant semi-transparent overlay with blur effect
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
    overlay.set_alpha(120)
    overlay.fill((248, 248, 248))
    screen.blit(overlay, (0, 0))
    
    # Modern menu card design
    menu_width = 380
    menu_height = 460  # Increased height to accommodate new options
    menu_x = (WINDOW_WIDTH - menu_width) // 2
    menu_y = (WINDOW_HEIGHT - menu_height) // 2
    
    # Menu shadow
    shadow_rect = pygame.Rect(menu_x + 4, menu_y + 4, menu_width, menu_height)
    pygame.draw.rect(screen, (0, 0, 0, 30), shadow_rect, border_radius=16)
    
    # Main menu background
    menu_rect = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
    pygame.draw.rect(screen, MENU_BOX_COLOR, menu_rect, border_radius=16)
    pygame.draw.rect(screen, PANEL_BORDER, menu_rect, 2, border_radius=16)
    
    # Modern typography
    title_font = pygame.font.Font(None, 36)
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)
    
    # Title with accent
    title_text = title_font.render("Settings", True, TEXT_COLOR)
    title_rect = title_text.get_rect(center=(menu_x + menu_width // 2, menu_y + 35))
    screen.blit(title_text, title_rect)
    
    # Title underline
    underline_rect = pygame.Rect(menu_x + 60, menu_y + 55, menu_width - 120, 2)
    pygame.draw.rect(screen, ACCENT_COLOR, underline_rect, border_radius=1)
    
    y_offset = 85
    item_height = 55
    
    # AI Difficulty setting
    difficulty_names = {1: "Beginner", 2: "Easy", 3: "Normal", 4: "Hard", 5: "Expert", 6: "Master"}
    difficulty_text = difficulty_names.get(ai_depth, "Custom")
    
    # Setting item background
    item_rect = pygame.Rect(menu_x + 20, menu_y + y_offset - 5, menu_width - 40, item_height - 10)
    pygame.draw.rect(screen, (248, 248, 248), item_rect, border_radius=8)
    
    ai_text = font.render(f"AI Difficulty", True, TEXT_COLOR)
    screen.blit(ai_text, (menu_x + 35, menu_y + y_offset))
    
    value_text = font.render(f"Level {ai_depth} ({difficulty_text})", True, ACCENT_COLOR)
    screen.blit(value_text, (menu_x + 35, menu_y + y_offset + 20))
    
    control_text = small_font.render("1 / 2", True, TEXT_SECONDARY)
    control_rect = control_text.get_rect(right=menu_x + menu_width - 35, centery=menu_y + y_offset + 15)
    screen.blit(control_text, control_rect)
    
    y_offset += item_height
    
    # Legal Moves setting
    item_rect = pygame.Rect(menu_x + 20, menu_y + y_offset - 5, menu_width - 40, item_height - 10)
    pygame.draw.rect(screen, (248, 248, 248), item_rect, border_radius=8)
    
    legal_text = font.render("Show Legal Moves", True, TEXT_COLOR)
    screen.blit(legal_text, (menu_x + 35, menu_y + y_offset))
    
    legal_status = "Enabled" if show_legal_moves else "Disabled"
    status_color = SUCCESS_COLOR if show_legal_moves else TEXT_SECONDARY
    status_text = font.render(legal_status, True, status_color)
    screen.blit(status_text, (menu_x + 35, menu_y + y_offset + 20))
    
    control_text = small_font.render("L", True, TEXT_SECONDARY)
    control_rect = control_text.get_rect(right=menu_x + menu_width - 35, centery=menu_y + y_offset + 15)
    screen.blit(control_text, control_rect)
    
    y_offset += item_height
    
    # Side Panel setting
    item_rect = pygame.Rect(menu_x + 20, menu_y + y_offset - 5, menu_width - 40, item_height - 10)
    pygame.draw.rect(screen, (248, 248, 248), item_rect, border_radius=8)
    
    panel_text = font.render("Side Panel", True, TEXT_COLOR)
    screen.blit(panel_text, (menu_x + 35, menu_y + y_offset))
    
    panel_status = "Visible" if show_side_panel else "Hidden"
    status_color = SUCCESS_COLOR if show_side_panel else TEXT_SECONDARY
    status_text = font.render(panel_status, True, status_color)
    screen.blit(status_text, (menu_x + 35, menu_y + y_offset + 20))
    
    control_text = small_font.render("P", True, TEXT_SECONDARY)
    control_rect = control_text.get_rect(right=menu_x + menu_width - 35, centery=menu_y + y_offset + 15)
    screen.blit(control_text, control_rect)
    
    y_offset += item_height
    
    # Switch Colors setting
    item_rect = pygame.Rect(menu_x + 20, menu_y + y_offset - 5, menu_width - 40, item_height - 10)
    pygame.draw.rect(screen, (248, 248, 248), item_rect, border_radius=8)
    
    color_text = font.render("Player Color", True, TEXT_COLOR)
    screen.blit(color_text, (menu_x + 35, menu_y + y_offset))
    
    color_status = "White" if player_is_white else "Black"
    status_color = TEXT_COLOR
    status_text = font.render(color_status, True, status_color)
    screen.blit(status_text, (menu_x + 35, menu_y + y_offset + 20))
    
    control_text = small_font.render("S", True, TEXT_SECONDARY)
    control_rect = control_text.get_rect(right=menu_x + menu_width - 35, centery=menu_y + y_offset + 15)
    screen.blit(control_text, control_rect)
    
    y_offset += item_height
    
    # Flip Board setting
    item_rect = pygame.Rect(menu_x + 20, menu_y + y_offset - 5, menu_width - 40, item_height - 10)
    pygame.draw.rect(screen, (248, 248, 248), item_rect, border_radius=8)
    
    flip_text = font.render("Board Orientation", True, TEXT_COLOR)
    screen.blit(flip_text, (menu_x + 35, menu_y + y_offset))
    
    flip_status = "Flipped" if board_flipped else "Normal"
    status_color = SUCCESS_COLOR if board_flipped else TEXT_SECONDARY
    status_text = font.render(flip_status, True, status_color)
    screen.blit(status_text, (menu_x + 35, menu_y + y_offset + 20))
    
    control_text = small_font.render("F", True, TEXT_SECONDARY)
    control_rect = control_text.get_rect(right=menu_x + menu_width - 35, centery=menu_y + y_offset + 15)
    screen.blit(control_text, control_rect)
    
    y_offset += item_height + 20  # Add some spacing before the close button
    
    # Close instruction at bottom
    close_rect = pygame.Rect(menu_x + 20, menu_y + y_offset, menu_width - 40, 30)
    pygame.draw.rect(screen, ACCENT_COLOR, close_rect, border_radius=8)
    
    close_text = font.render("Press M to close", True, (255, 255, 255))
    close_text_rect = close_text.get_rect(center=close_rect.center)
    screen.blit(close_text, close_text_rect)

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
    
    # Minimalistic QoL improvements
    auto_save_enabled = True
    show_move_timer = False  # Start disabled for cleaner look
    highlight_last_move = True
    show_captured_pieces = False  # Start disabled for minimalism
    enable_right_click_deselect = True
    show_coordinates = False
    enable_drag_drop = True
    smooth_animations = True
    subtle_hover_effects = True
    
    # Color and board orientation settings
    player_is_white = True  # True = player is white, False = player is black
    board_flipped = False   # True = board is flipped (black perspective)
    
    # Drag and drop state
    dragging_piece = None
    drag_start_pos = None
    drag_offset = (0, 0)
    
    # Game state tracking
    game_start_time = pygame.time.get_ticks()
    last_move_time = game_start_time
    captured_white = []  # Pieces captured by black
    captured_black = []  # Pieces captured by white
    hovered_square = None  # For subtle hover effects
    
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
                    captured_white = []
                    captured_black = []
                    game_start_time = current_time
                    last_move_time = current_time
                elif event.key == pygame.K_c and not show_menu:
                    # Toggle coordinates
                    show_coordinates = not show_coordinates
                elif event.key == pygame.K_t and not show_menu:
                    # Toggle move timer
                    show_move_timer = not show_move_timer
                elif event.key == pygame.K_v and not show_menu:
                    # Toggle captured pieces display
                    show_captured_pieces = not show_captured_pieces
                elif event.key == pygame.K_SPACE and not show_menu:
                    # Deselect piece
                    selected_square = None
                    legal_moves = []
                elif event.key == pygame.K_f and not show_menu:
                    # Flip board (rotate view)
                    board_flipped = not board_flipped
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
                    elif event.key == pygame.K_s:
                        # Switch player/AI colors
                        player_is_white = not player_is_white
                        # Reset game when switching colors
                        if game_moves:
                            player_model.record_game(game_moves, completed=board.is_game_over())
                        board.reset()
                        selected_square = None
                        legal_moves = []
                        move_history = []
                        last_move = None
                        game_moves = []
                        captured_white = []
                        captured_black = []
                        game_start_time = current_time
                        last_move_time = current_time
                    elif event.key == pygame.K_f:
                        # Flip board
                        board_flipped = not board_flipped
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    esc_hold_start = None
                    esc_held = False
            
            elif event.type == pygame.MOUSEMOTION:
                # Handle hover effects and dragging
                if subtle_hover_effects and not show_menu:
                    mouse_square = get_square_from_pos(event.pos, board_flipped)
                    hovered_square = mouse_square if mouse_square is not None else None
                
                # Handle piece dragging
                if dragging_piece and enable_drag_drop:
                    drag_offset = (event.pos[0] - drag_start_pos[0], 
                                 event.pos[1] - drag_start_pos[1])
            
            elif event.type == pygame.MOUSEBUTTONDOWN and not show_menu and ((player_is_white and board.turn == chess.WHITE) or (not player_is_white and board.turn == chess.BLACK)) and not board.is_game_over():
                if event.button == 3 and enable_right_click_deselect:  # Right click
                    # Right click to deselect
                    selected_square = None
                    legal_moves = []
                elif event.button == 1:  # Left click
                    # Calculate mouse position relative to board
                    mouse_x, mouse_y = event.pos
                    current_width = screen.get_width()
                
                    if show_side_panel:
                        # Normal layout - board starts at (0, 0)
                        square = get_square_from_pos(event.pos, board_flipped)
                    else:
                        # Board fills entire window when side panel is off
                        square = get_square_from_pos(event.pos, board_flipped)
                    
                    if square is not None:
                        if selected_square is None:
                            # Select piece and potentially start drag
                            piece = board.piece_at(square)
                            player_color = chess.WHITE if player_is_white else chess.BLACK
                            if piece and piece.color == player_color:
                                selected_square = square
                                if show_legal_moves:
                                    legal_moves = get_legal_moves_for_square(board, square)
                                
                                # Initialize drag state
                                if enable_drag_drop:
                                    dragging_piece = piece
                                    drag_start_pos = event.pos
                                    drag_offset = (0, 0)
                        else:
                            # Make move and track captures
                            captured_piece = board.piece_at(square)
                            move = chess.Move(selected_square, square)
                            if move in board.legal_moves:
                                # Track captured pieces
                                if captured_piece:
                                    if captured_piece.color == chess.BLACK:
                                        captured_black.append(captured_piece.symbol())
                                    else:
                                        captured_white.append(captured_piece.symbol())
                                
                                board.push(move)
                                move_history.append(move.uci())
                                game_moves.append(move.uci())
                                last_move = move
                                last_move_time = current_time
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
            
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                # Handle drag and drop completion
                if dragging_piece and enable_drag_drop:
                    target_square = get_square_from_pos(event.pos, board_flipped)
                    if target_square is not None and selected_square is not None:
                        # Try to make the move
                        captured_piece = board.piece_at(target_square)
                        move = chess.Move(selected_square, target_square)
                        if move in board.legal_moves:
                            # Track captured pieces
                            if captured_piece:
                                if captured_piece.color == chess.BLACK:
                                    captured_black.append(captured_piece.symbol())
                                else:
                                    captured_white.append(captured_piece.symbol())
                            
                            board.push(move)
                            move_history.append(move.uci())
                            game_moves.append(move.uci())
                            last_move = move
                            last_move_time = current_time
                            
                            # Start AI thinking
                            if not board.is_game_over():
                                ai.start_thinking(board)
                
                # Reset drag state
                dragging_piece = None
                drag_start_pos = None
                drag_offset = (0, 0)
                selected_square = None
                legal_moves = []
        
        # Check ESC hold
        if esc_hold_start and not esc_held:
            if current_time - esc_hold_start >= esc_hold_duration:
                esc_held = True
                if game_moves:
                    player_model.record_game(game_moves, completed=board.is_game_over())
                running = False
        
        # AI move - AI plays the opposite color from player
        ai_color = chess.BLACK if player_is_white else chess.WHITE
        if board.turn == ai_color and not board.is_game_over() and not ai.thinking:
            ai_move = ai.get_move()
            if ai_move:
                # Track AI captures
                captured_piece = board.piece_at(ai_move.to_square)
                player_color = chess.WHITE if player_is_white else chess.BLACK
                if captured_piece and captured_piece.color == player_color:
                    if player_color == chess.WHITE:
                        captured_white.append(captured_piece.symbol())
                    else:
                        captured_black.append(captured_piece.symbol())
                
                board.push(ai_move)
                move_history.append(ai_move.uci())
                game_moves.append(ai_move.uci())
                last_move = ai_move
                last_move_time = current_time
            elif not ai.thinking:
                ai.start_thinking(board)
        
        # Draw everything
        screen.fill(BACKGROUND_COLOR)
        
        # Calculate board position based on window size
        current_width = screen.get_width()
        current_height = screen.get_height()
        
        if show_side_panel:
            # Normal layout with side panel
            draw_board(screen, board, images, selected_square, legal_moves if show_legal_moves else None, 
                      last_move, hovered_square, dragging_piece, drag_offset, board_flipped)
        else:
            # Full window board when side panel is off
            draw_board(screen, board, images, selected_square, legal_moves if show_legal_moves else None, 
                      last_move, hovered_square, dragging_piece, drag_offset, board_flipped)
        
        # Draw board coordinates
        draw_coordinates(screen, show_coordinates, board_flipped)
        
        # Draw side panel only if enabled
        if show_side_panel:
            draw_panel(screen, board, ai, move_history, captured_white, captured_black, 
                      show_move_timer, last_move_time, current_time, show_captured_pieces, show_coordinates)
        
        # Draw ESC hold progress
        if esc_hold_start and not esc_held:
            # Simple minimalist text only - centered on screen
            font = pygame.font.Font(None, 24)
            text = font.render("Hold ESC to quit", True, TEXT_COLOR)
            text_rect = text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            screen.blit(text, text_rect)
        
        # Draw menu overlay if active
        if show_menu:
            draw_menu_overlay(screen, ai.depth, show_legal_moves, show_side_panel, player_is_white, board_flipped)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
