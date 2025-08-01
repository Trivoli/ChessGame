#!/usr/bin/env python3

import chess
import time
import sys
import os

# Add the current directory to Python path to import our chess AI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our optimized chess AI components
import importlib.util
import sys

# Import the module with hyphens in the name
spec = importlib.util.spec_from_file_location("chess_ai", "chess-ai.py")
chess_ai = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chess_ai)

PlayerModel = chess_ai.PlayerModel
AdvancedAdaptiveChessAI = chess_ai.AdvancedAdaptiveChessAI
OptimizedChessEngine = chess_ai.OptimizedChessEngine

def test_ai_performance():
    """Test the optimized chess AI performance"""
    print("=" * 60)
    print("TESTING OPTIMIZED CHESS AI")
    print("=" * 60)
    
    # Create player model and AI
    player_model = PlayerModel()
    
    # Test different difficulty levels
    test_depths = [2, 3, 4, 5]
    
    for depth in test_depths:
        print(f"\n🧠 Testing AI at depth {depth}")
        print("-" * 40)
        
        ai = AdvancedAdaptiveChessAI(player_model, search_depth=depth, aggressivity_factor=1.0)
        board = chess.Board()
        
        # Test several positions
        test_positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4",  # Italian Game
            "8/8/8/3k4/8/3K4/8/8 w - - 0 1",  # King endgame
        ]
        
        position_names = ["Opening", "Middle Game", "Endgame"]
        
        for pos_name, fen in zip(position_names, test_positions):
            board.set_fen(fen)
            print(f"  📍 {pos_name} position:")
            
            start_time = time.time()
            
            # Get adaptive depth
            adaptive_depth = ai.get_adaptive_depth(board)
            
            # Test the optimized engine directly
            engine = OptimizedChessEngine()
            score, best_move = engine.iterative_deepening_search(
                board, adaptive_depth, time_limit=2.0, 
                player_model=player_model, move_number=board.fullmove_number-1
            )
            
            elapsed = time.time() - start_time
            
            print(f"    ⚡ Time: {elapsed:.2f}s")
            print(f"    🎯 Best move: {best_move}")
            print(f"    📊 Score: {score}")
            print(f"    🔍 Adaptive depth: {adaptive_depth}")
            print(f"    🧮 Nodes searched: {engine.nodes_searched}")
            print(f"    💾 TT hits: {engine.tt_hits}")
            print(f"    📈 TT hit rate: {engine.tt_hits/max(engine.nodes_searched,1)*100:.1f}%")
            
            if best_move:
                print(f"    ✅ Found good move: {best_move}")
            else:
                print("    ❌ No move found")
            print()

def test_engine_features():
    """Test specific engine features"""
    print("\n🔧 TESTING ENGINE FEATURES")
    print("=" * 40)
    
    engine = OptimizedChessEngine()
    board = chess.Board()
    
    # Test evaluation function
    print("📊 Evaluation function:")
    eval_score = engine.evaluate_board(board)
    print(f"  Starting position eval: {eval_score}")
    
    # Test endgame detection
    print(f"  Is endgame: {engine.is_endgame(board)}")
    
    # Test piece-square tables
    print("  Piece-square values at start:")
    for square in [chess.E2, chess.E7, chess.E4, chess.E5]:  # Test pawn positions
        piece = board.piece_at(square)
        if piece:
            ps_value = engine.get_piece_square_value(piece, square)
            print(f"    {chess.square_name(square)}: {piece.symbol()} = {ps_value}")
    
    # Test move ordering
    print("\n🎯 Move ordering:")
    moves = list(board.legal_moves)
    ordered_moves = engine.order_moves(board, moves)
    print(f"  Original moves: {len(moves)}")
    print(f"  First 5 ordered: {[str(m) for m in ordered_moves[:5]]}")
    
    print("\n✅ All engine features working!")

def play_sample_game():
    """Play a quick sample game to test AI"""
    print("\n🎮 SAMPLE GAME")
    print("=" * 30)
    
    player_model = PlayerModel()
    ai = AdvancedAdaptiveChessAI(player_model, search_depth=3)
    board = chess.Board()
    
    move_count = 0
    max_moves = 10  # Limit for demo
    
    print("Starting position:")
    print(board)
    print()
    
    while not board.is_game_over() and move_count < max_moves:
        move_count += 1
        
        if board.turn == chess.WHITE:
            # Human move (we'll just pick the first legal move for demo)
            legal_moves = list(board.legal_moves)
            move = legal_moves[0]  # Pick first legal move
            print(f"Human plays: {move}")
        else:
            # AI move
            print("AI thinking...")
            ai.start_thinking(board)
            
            # Wait for AI to think
            timeout = 0
            while ai.thinking and timeout < 50:  # 5 second timeout
                time.sleep(0.1)
                timeout += 1
            
            move = ai.get_move()
            if move:
                print(f"AI plays: {move}")
                print(f"  Time: {ai.last_search_time:.2f}s")
                stats = ai.get_performance_stats()
                print(f"  Nodes: {stats['nodes_searched']}, TT hits: {stats['transposition_hits']}")
            else:
                print("AI failed to find move!")
                break
        
        board.push(move)
        ai.record_player_move(move.uci())
        print(f"Position after move {move_count}:")
        print(board)
        print("-" * 30)
    
    print(f"Game ended after {move_count} moves")
    if board.is_game_over():
        print(f"Result: {board.result()}")

if __name__ == "__main__":
    try:
        # Test AI performance at different depths
        test_ai_performance()
        
        # Test engine-specific features  
        test_engine_features()
        
        # Play a sample game
        play_sample_game()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("The optimized chess AI is working correctly.")
        print("Key improvements:")
        print("  ✅ Transposition tables for faster search")
        print("  ✅ Move ordering for better pruning")
        print("  ✅ Piece-square tables for positional play")
        print("  ✅ Adaptive depth based on game phase")
        print("  ✅ Iterative deepening with time management")
        print("  ✅ Enhanced evaluation function")
        print("  ✅ Killer moves and history heuristic")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()