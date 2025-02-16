from service import ask_bot

if __name__ == "__main__":
    try:
        print("\nStarting Islamic Fatwa Assistant...")
        # Initialize the query engine
        print("\nIslamic Fatwa Assistant (Press Ctrl+C to exit)")
        print("----------------------------------------")
        
        while True:
            try:
                # Get user input
                question = input("\nYour question: ").strip()
                
                # Check for empty input
                if not question:
                    continue
                    
                # Check for exit commands
                if question.lower() in ['exit', 'quit', 'q']:
                    print("\nThank you for using the Islamic Fatwa Assistant. Goodbye!")
                    break
                
                # Get response
                print("\nSearching for answer...")
                response = ask_bot(question)
                if isinstance(response, dict):
                    print(response.get('output', 'No response generated'))
                else:
                    print(response)
                print("\n----------------------------------------")
                
            except Exception as e:
                print(f"\nError processing question: {str(e)}")
                print("Please try again.")
                continue
                
    except KeyboardInterrupt:
        print("\n\nThank you for using the Islamic Fatwa Assistant. Goodbye!")
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        print("Please check your configuration and try again.") 