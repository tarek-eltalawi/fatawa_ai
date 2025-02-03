import warnings
from QA_langgraph import ask_bot
from agents.islamic_agent import create_islamic_agent

# Suppress all LangChain warnings
warnings.filterwarnings('ignore', category=UserWarning, module='langchain')

if __name__ == "__main__":
    try:
        print("\nStarting Islamic Fatwa Assistant...")
        # Initialize the query engine
        agent = create_islamic_agent()
        print("Agent created successfully")
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
                print("Invoking agent with question:", question)
                # response = agent.invoke({"input": question})
                response = ask_bot(question)
                print("\nAgent response received")
                if isinstance(response, dict):
                    print(response.get('output', 'No response generated'))
                else:
                    print(response)
                print("\n----------------------------------------")
                
            except Exception as e:
                print(f"\nError processing question: {str(e)}")
                print("Please try again with a different question.")
                continue
                
    except KeyboardInterrupt:
        print("\n\nThank you for using the Islamic Fatwa Assistant. Goodbye!")
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        print("Please check your configuration and try again.") 