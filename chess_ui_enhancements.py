#!/usr/bin/env python3
"""
Chess UI Enhancements
Additional UI components and improvements for the chess game
"""

import pygame
import math
import time
from typing import Tuple, List, Optional

# Additional UI constants for enhancements
WELCOME_SCREEN_DURATION = 3000  # 3 seconds
NOTIFICATION_DURATION = 2000    # 2 seconds
GLOW_ANIMATION_SPEED = 0.05
PULSE_ANIMATION_SPEED = 0.03

class NotificationSystem:
    """Enhanced notification system for game events"""
    
    def __init__(self):
        self.notifications = []
        self.font = None
    
    def add_notification(self, message: str, color: Tuple[int, int, int] = (34, 197, 94), duration: int = NOTIFICATION_DURATION):
        """Add a new notification to display"""
        notification = {
            'message': message,
            'color': color,
            'start_time': pygame.time.get_ticks(),
            'duration': duration,
            'alpha': 255
        }
        self.notifications.append(notification)
    
    def update(self):
        """Update notification states and remove expired ones"""
        current_time = pygame.time.get_ticks()
        self.notifications = [n for n in self.notifications if current_time - n['start_time'] < n['duration']]
        
        # Update alpha for fade-out effect
        for notification in self.notifications:
            elapsed = current_time - notification['start_time']
            fade_start = notification['duration'] * 0.7  # Start fading at 70% of duration
            if elapsed > fade_start:
                fade_progress = (elapsed - fade_start) / (notification['duration'] - fade_start)
                notification['alpha'] = int(255 * (1 - fade_progress))
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw all active notifications"""
        if not self.notifications:
            return
        
        y_offset = 10
        for notification in self.notifications:
            # Create text surface with alpha
            text_surface = font.render(notification['message'], True, notification['color'])
            text_surface.set_alpha(notification['alpha'])
            
            # Create background with alpha
            text_rect = text_surface.get_rect()
            bg_rect = pygame.Rect(10, y_offset, text_rect.width + 20, text_rect.height + 10)
            bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surface.fill((*notification['color'], 40))  # Semi-transparent background
            
            screen.blit(bg_surface, bg_rect)
            screen.blit(text_surface, (20, y_offset + 5))
            
            y_offset += bg_rect.height + 5

class WelcomeScreen:
    """Animated welcome screen for the chess game"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.start_time = pygame.time.get_ticks()
        self.logo_scale = 0.0
        self.text_alpha = 0
        
    def update(self):
        """Update animation states"""
        elapsed = pygame.time.get_ticks() - self.start_time
        
        # Logo scale animation
        if elapsed < 1000:
            self.logo_scale = min(1.0, elapsed / 1000.0)
        else:
            self.logo_scale = 1.0
        
        # Text fade-in animation
        if elapsed > 500 and elapsed < 2000:
            self.text_alpha = min(255, int((elapsed - 500) * 255 / 1500))
        elif elapsed >= 2000:
            self.text_alpha = 255
    
    def draw(self, screen: pygame.Surface, fonts: dict):
        """Draw the welcome screen"""
        # Dark gradient background
        for y in range(self.height):
            color_value = int(20 + (y / self.height) * 25)
            pygame.draw.line(screen, (color_value, color_value, color_value + 5), (0, y), (self.width, y))
        
        # Main title with scale animation
        title_font = fonts.get('title', pygame.font.SysFont('Arial', 48, bold=True))
        title_text = title_font.render('ChesslerAI', True, (248, 250, 252))
        
        # Apply scaling
        if self.logo_scale < 1.0:
            scaled_width = int(title_text.get_width() * self.logo_scale)
            scaled_height = int(title_text.get_height() * self.logo_scale)
            title_text = pygame.transform.scale(title_text, (scaled_width, scaled_height))
        
        title_rect = title_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
        screen.blit(title_text, title_rect)
        
        # Subtitle with fade-in
        if self.text_alpha > 0:
            subtitle_font = fonts.get('subtitle', pygame.font.SysFont('Arial', 24))
            subtitle_text = subtitle_font.render('Advanced Chess AI with Adaptive Learning', True, (200, 200, 200))
            subtitle_text.set_alpha(self.text_alpha)
            subtitle_rect = subtitle_text.get_rect(center=(self.width // 2, self.height // 2 + 10))
            screen.blit(subtitle_text, subtitle_rect)
            
            # Loading text
            loading_text = subtitle_font.render('Loading...', True, (150, 150, 150))
            loading_text.set_alpha(self.text_alpha)
            loading_rect = loading_text.get_rect(center=(self.width // 2, self.height // 2 + 60))
            screen.blit(loading_text, loading_rect)
    
    def is_finished(self) -> bool:
        """Check if welcome screen should end"""
        return pygame.time.get_ticks() - self.start_time > WELCOME_SCREEN_DURATION

class EnhancedButton:
    """Enhanced button with hover effects and animations"""
    
    def __init__(self, x: int, y: int, width: int, height: int, text: str, 
                 color: Tuple[int, int, int] = (51, 65, 85),
                 hover_color: Tuple[int, int, int] = (71, 85, 105),
                 text_color: Tuple[int, int, int] = (248, 250, 252)):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.is_hovered = False
        self.glow_alpha = 0
        self.press_animation = 0
        
    def update(self, mouse_pos: Tuple[int, int]):
        """Update button state based on mouse position"""
        self.is_hovered = self.rect.collidepoint(mouse_pos)
        
        # Glow animation
        if self.is_hovered:
            self.glow_alpha = min(255, self.glow_alpha + 5)
        else:
            self.glow_alpha = max(0, self.glow_alpha - 5)
            
        # Reset press animation
        if self.press_animation > 0:
            self.press_animation = max(0, self.press_animation - 0.1)
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Draw the enhanced button"""
        # Glow effect
        if self.glow_alpha > 0:
            glow_surface = pygame.Surface((self.rect.width + 10, self.rect.height + 10), pygame.SRCALPHA)
            glow_color = (*self.hover_color, self.glow_alpha // 3)
            pygame.draw.rect(glow_surface, glow_color, 
                           (0, 0, self.rect.width + 10, self.rect.height + 10), 
                           border_radius=8)
            screen.blit(glow_surface, (self.rect.x - 5, self.rect.y - 5))
        
        # Button background with press animation
        button_rect = self.rect.copy()
        if self.press_animation > 0:
            offset = int(2 * self.press_animation)
            button_rect.x += offset
            button_rect.y += offset
        
        current_color = self.hover_color if self.is_hovered else self.color
        pygame.draw.rect(screen, current_color, button_rect, border_radius=6)
        
        # Button text
        text_surface = font.render(self.text, True, self.text_color)
        text_rect = text_surface.get_rect(center=button_rect.center)
        screen.blit(text_surface, text_rect)
    
    def handle_click(self) -> bool:
        """Handle button click and return True if clicked"""
        if self.is_hovered:
            self.press_animation = 1.0
            return True
        return False

class GameModeSelector:
    """Enhanced game mode selection screen"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.buttons = []
        self.setup_buttons()
        
    def setup_buttons(self):
        """Setup game mode selection buttons"""
        button_width = 200
        button_height = 50
        center_x = self.width // 2 - button_width // 2
        start_y = self.height // 2 - 100
        
        modes = [
            ("Play vs AI", "play_ai"),
            ("Watch AI Demo", "demo"),
            ("Settings", "settings"),
            ("Exit", "exit")
        ]
        
        for i, (text, mode) in enumerate(modes):
            button = EnhancedButton(center_x, start_y + i * 70, button_width, button_height, text)
            button.mode = mode
            self.buttons.append(button)
    
    def update(self, mouse_pos: Tuple[int, int]) -> Optional[str]:
        """Update buttons and return selected mode if any"""
        for button in self.buttons:
            button.update(mouse_pos)
        return None
    
    def handle_click(self, mouse_pos: Tuple[int, int]) -> Optional[str]:
        """Handle button clicks and return selected mode"""
        for button in self.buttons:
            if button.handle_click():
                return button.mode
        return None
    
    def draw(self, screen: pygame.Surface, fonts: dict):
        """Draw the game mode selection screen"""
        # Background
        screen.fill((15, 23, 42))
        
        # Title
        title_font = fonts.get('title', pygame.font.SysFont('Arial', 36, bold=True))
        title_text = title_font.render('ChesslerAI', True, (248, 250, 252))
        title_rect = title_text.get_rect(center=(self.width // 2, 100))
        screen.blit(title_text, title_rect)
        
        # Subtitle
        subtitle_font = fonts.get('subtitle', pygame.font.SysFont('Arial', 18))
        subtitle_text = subtitle_font.render('Select Game Mode', True, (200, 200, 200))
        subtitle_rect = subtitle_text.get_rect(center=(self.width // 2, 140))
        screen.blit(subtitle_text, subtitle_rect)
        
        # Draw buttons
        button_font = fonts.get('button', pygame.font.SysFont('Arial', 16, bold=True))
        for button in self.buttons:
            button.draw(screen, button_font)

def create_enhanced_fonts() -> dict:
    """Create enhanced font dictionary for UI"""
    return {
        'title': pygame.font.SysFont('Arial', 48, bold=True),
        'subtitle': pygame.font.SysFont('Arial', 24),
        'button': pygame.font.SysFont('Arial', 16, bold=True),
        'notification': pygame.font.SysFont('Arial', 14)
    }

# Example of how to integrate these enhancements:
"""
# In your main game loop, you would add:

# Initialize enhancements
notification_system = NotificationSystem()
enhanced_fonts = create_enhanced_fonts()

# Show welcome screen
welcome = WelcomeScreen(WIDTH, HEIGHT)
game_state = "welcome"

# In game loop:
if game_state == "welcome":
    welcome.update()
    welcome.draw(screen, enhanced_fonts)
    if welcome.is_finished():
        game_state = "menu"

# Add notifications for game events:
notification_system.add_notification("Check!", (251, 191, 36))  # Yellow for check
notification_system.add_notification("Checkmate!", (239, 68, 68))  # Red for checkmate
notification_system.add_notification("Good move!", (34, 197, 94))  # Green for good moves

# Update and draw notifications
notification_system.update()
notification_system.draw(screen, enhanced_fonts['notification'])
"""