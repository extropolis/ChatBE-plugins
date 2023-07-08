from .local_search import LocalSearchTool

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    questions = ["What are some good restaurants around here?",
                 "What is the best Thai restaurant that's open past 9pm?",
                 "What's the weather today?",
                 "Is it going to rain tomorrow?",
                 "Is it going to be nice weather this week?",
                 "What's the closest national park?",
                 "What's the closest state park?",
                 "Where is the closest camping ground?",
                 "Where is the closest hotel?",
                 "Where is the closest RV dump station?"]
    ls_tool = LocalSearchTool(user_id="Winston")
    print(ls_tool.to_json())
    print(ls_tool.run({"query": questions[0]}))
    print(ls_tool.run({"query": questions[-1]}))