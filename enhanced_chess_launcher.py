#!/usr/bin/env python3
"""
Enhanced Chess Game Launcher
Combines the existing sophisticated chess AI with additional UI enhancements
"""

import pygame
import sys
import os
import subprocess
from chess_ui_enhancements import (
    NotificationSystem, WelcomeScreen, GameModeSelector, 
    EnhancedButton, create_enhanced_fonts
)

# Import chess AI components
import importlib.util
spec = importlib.util.spec_from_file_location("chess_ai", "chess-ai.py")
chess_ai = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chess_ai)

class EnhancedChessLauncher:
    """Enhanced launcher for the chess game with additional UI features"""
    
    def __init__(self):
        pygame.init()
        
        # Use the same dimensions as the main game
        self.width = chess_ai.WIDTH
        self.height = chess_ai.HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('ChesslerAI - Enhanced Edition')
        
        # Setup components
        self.clock = pygame.time.Clock()
        self.fonts = create_enhanced_fonts()
        self.notification_system = NotificationSystem()
        self.welcome_screen = WelcomeScreen(self.width, self.height)
        self.game_mode_selector = GameModeSelector(self.width, self.height)
        
        # Game state
        self.state = "welcome"  # welcome -> menu -> game
        self.running = True
        
        # Add welcome notification
        self.notification_system.add_notification(
            "Welcome to ChesslerAI Enhanced Edition!", 
            (34, 197, 94), 
            4000
        )
    
    def handle_events(self):
        """Handle pygame events"""
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state == "menu":
                        self.state = "welcome"
                    else:
                        self.running = False
                elif event.key == pygame.K_SPACE and self.state == "welcome":
                    # Skip welcome screen
                    self.state = "menu"
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.state == "menu":
                    mode = self.game_mode_selector.handle_click(mouse_pos)
                    if mode:
                        self.handle_mode_selection(mode)
        
        # Update components
        if self.state == "menu":
            self.game_mode_selector.update(mouse_pos)
    
    def handle_mode_selection(self, mode: str):
        """Handle game mode selection"""
        if mode == "play_ai":
            self.notification_system.add_notification("Starting Chess Game...", (59, 130, 246))
            pygame.time.wait(1000)  # Brief pause to show notification
            self.launch_chess_game()
        elif mode == "demo":
            self.notification_system.add_notification("AI Demo Mode not implemented yet", (251, 191, 36))
        elif mode == "settings":
            self.notification_system.add_notification("Settings panel not implemented yet", (251, 191, 36))
        elif mode == "exit":
            self.running = False
    
    def launch_chess_game(self):
        """Launch the main chess game"""
        pygame.quit()
        try:
            # Run the main chess game
            subprocess.run([sys.executable, "chess-ai.py"])
        except Exception as e:
            print(f"Error launching chess game: {e}")
        finally:
            # Restart the launcher after chess game ends
            self.__init__()
    
    def update(self):
        """Update game components"""
        if self.state == "welcome":
            self.welcome_screen.update()
            if self.welcome_screen.is_finished():
                self.state = "menu"
                self.notification_system.add_notification("Select a game mode", (200, 200, 200))
        
        self.notification_system.update()
    
    def draw(self):
        """Draw the current screen"""
        if self.state == "welcome":
            self.welcome_screen.draw(self.screen, self.fonts)
            
            # Add skip instruction
            skip_font = self.fonts['notification']
            skip_text = skip_font.render("Press SPACE to skip", True, (150, 150, 150))
            skip_rect = skip_text.get_rect(bottomright=(self.width - 20, self.height - 20))
            self.screen.blit(skip_text, skip_rect)
            
        elif self.state == "menu":
            self.game_mode_selector.draw(self.screen, self.fonts)
            
            # Add instructions
            instruction_font = self.fonts['notification']
            instruction_text = instruction_font.render("ESC: Back to Welcome | Click to select", True, (150, 150, 150))
            instruction_rect = instruction_text.get_rect(center=(self.width // 2, self.height - 30))
            self.screen.blit(instruction_text, instruction_rect)
        
        # Always draw notifications on top
        self.notification_system.draw(self.screen, self.fonts['notification'])
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        print("🚀 ChesslerAI Enhanced Edition Starting...")
        print("✨ Features:")
        print("   - Animated welcome screen")
        print("   - Enhanced menu system")
        print("   - Notification system")
        print("   - Smooth UI animations")
        print("   - All original chess features preserved")
        print("\n🎮 Use mouse to navigate, ESC to go back, SPACE to skip animations")
        
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
        print("👋 Thanks for playing ChesslerAI!")

def main():
    """Main entry point"""
    # Check if required files exist
    required_files = ["chess-ai.py", "chess_ui_enhancements.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return
    
    # Check if piece images exist
    piece_dir = "Pawns"
    if not os.path.exists(piece_dir):
        print(f"⚠️  Warning: {piece_dir} directory not found. Pieces will use fallback rendering.")
    
    try:
        launcher = EnhancedChessLauncher()
        launcher.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()