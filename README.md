# 📘 Sổ tay LangGraph Academy - Module 1: Các Khái Niệm Cơ Bản

Module này giúp chuyển đổi tư duy từ việc viết code tuần tự sang xây dựng luồng làm việc (workflow) dạng đồ thị cho AI Agent bằng LangGraph. Dưới đây là 4 thành phần cốt lõi.

---

## 1. Cấu trúc Tin nhắn (Messages)

Các mô hình ngôn ngữ (LLM) mặc định **không có trí nhớ**. Để AI hiểu ngữ cảnh, ta phải gửi toàn bộ lịch sử hội thoại dưới dạng danh sách các tin nhắn đã được "dán nhãn" vai trò.
```python
from pprint import pprint
from langchain_core.messages import AIMessage, HumanMessage

# Khởi tạo lịch sử trò chuyện
messages = [AIMessage(content="So you said you were researching ocean mammals?", name="Model")]
messages.append(HumanMessage(content="Yes, that's right.", name="Lance"))
messages.append(AIMessage(content="Great, what would you like to learn about.", name="Model"))
messages.append(HumanMessage(content="I want to learn about the best place to see Orcas in the US.", name="Lance"))

for m in messages:
    m.pretty_print()
```

> 💡 **Khác biệt giữa `HumanMessage` và `AIMessage`:**
>
> - **Dịch vai trò:** LangChain sẽ tự động dịch `HumanMessage` thành vai trò `user` và `AIMessage` thành vai trò `assistant` khi gọi API của LLM.
> - **Dữ liệu ẩn:** Chỉ `AIMessage` mới có khả năng chứa thuộc tính `tool_calls` (khi AI quyết định sử dụng công cụ).
> - **Điều hướng (Routing):** LangGraph thường nhìn vào loại tin nhắn cuối cùng để quyết định bước đi tiếp theo trong đồ thị.

---

## 2. Gọi Công cụ (Tool Calling)

Biến một Chatbot bình thường thành một Agent có khả năng hành động bằng cách trang bị công cụ cho nó.
```python
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# Trang bị công cụ cho LLM
llm_with_tools = llm.bind_tools([multiply])
```

> 💡 **Lưu ý quan trọng:**
>
> - **Type hints** (`a: int`) và **Docstring** (`"""..."""`) là bắt buộc! AI không đọc code Python bên trong hàm — nó chỉ đọc Docstring để hiểu công cụ này dùng làm gì và khi nào nên dùng.
> - Hàm `.bind_tools()` không chạy công cụ, nó chỉ "dịch" công cụ thành định dạng JSON và đưa cho AI quyển hướng dẫn sử dụng.

---

## 3. Trí nhớ chung (State & Reducers)

LangGraph là một **cỗ máy trạng thái** (state machine). Cần một bộ nhớ chung để các nút (Nodes) đọc và ghi vào.
```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# Cách 1: Tự định nghĩa State
class MessagesState(TypedDict):
    # Dùng Annotated và add_messages để "Cộng dồn" thay vì "Ghi đè"
    messages: Annotated[list[AnyMessage], add_messages]

# Cách 2: Đường tắt (Sử dụng class tích hợp sẵn của LangGraph)
from langgraph.graph import MessagesState

class CustomMessagesState(MessagesState):
    # Có thể thêm các biến bộ nhớ khác ở đây nếu cần
    pass
```

> 💡 **Vai trò của `add_messages`:**
>
> Đây là một **hàm thu gọn (Reducer)**. Khi một Node trả về tin nhắn mới, `add_messages` đảm bảo tin nhắn đó được nối tiếp vào cuối danh sách lịch sử cũ, giúp Agent không bị mất trí nhớ.

---

## 4. Xây dựng Đồ thị (Building the Graph)

Lắp ráp tất cả các mảnh ghép (LLM, Tools, State) thành một workflow hoàn chỉnh.
```python
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END

# 1. Định nghĩa Node (Nút xử lý)
def tool_calling_llm(state: CustomMessagesState):
    # Lấy lịch sử hiện tại, gọi AI, và trả về dict chứa tin nhắn mới để cập nhật State
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 2. Xây dựng đồ thị
builder = StateGraph(CustomMessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)  # Thêm Nút
builder.add_edge(START, "tool_calling_llm")             # Nối từ Bắt đầu đến Nút
builder.add_edge("tool_calling_llm", END)               # Nối từ Nút đến Kết thúc

# 3. Đóng gói Agent
graph = builder.compile()

# (Tùy chọn) Vẽ sơ đồ ra màn hình
# display(Image(graph.get_graph().draw_mermaid_png()))
```

---
---

# 📘 Sổ Tay LangGraph - Phần 2: Từ Chatbot đến Tác tử AI tự trị (Agent)

---

## 5. Kiến trúc ReAct (Trái tim của AI Agent)

Sự khác biệt lớn nhất giữa một **Router** (chỉ biết định tuyến 1 lần) và một **Agent** (tác tử tự trị) nằm ở **Vòng lặp (Loop)**. Kiến trúc này được gọi là **ReAct** (Reason + Act).

- **Reason (Suy luận):** AI suy nghĩ xem cần dùng công cụ gì.
- **Act (Hành động):** AI gọi công cụ.
- **Observe (Quan sát):** Kết quả từ công cụ được **trả ngược lại** cho AI để nó tiếp tục suy nghĩ và hành động, cho đến khi hoàn thành nhiệm vụ.
```python
# CODE CŨ (Router): Chạy tool xong là kết thúc
builder.add_edge("tools", END)

# CODE MỚI (Agent): Chạy tool xong, cầm kết quả quay về nộp cho AI
builder.add_edge("tools", "assistant")
```

---

## 6. Tinh chỉnh Mô hình AI (Prompt & Config)

### 6.1. Khắc phục lỗi "Gọi tool song song" (Parallel Tool Calls)

Mặc định, các AI xịn (như OpenAI) có khả năng gọi nhiều tool cùng lúc. Nhưng đôi khi ta muốn nó chạy tuần tự từng bước một (ví dụ: tìm tên CEO xong mới được gửi email).

- **Với OpenAI:** Dùng `bind_tools(tools, parallel_tool_calls=False)`.
- **Với Google Gemini:** Bỏ tham số này đi vì cơ chế API khác nhau, chỉ cần `bind_tools(tools)`.

### 6.2. Gắn "Nội quy" bằng `SystemMessage`

Để AI không bị lan man, ta ghim một câu lệnh định hướng lên trên cùng của lịch sử chat trước khi đưa cho AI đọc.
```python
from langchain_core.messages import SystemMessage

sys_msg = SystemMessage(content="Bạn là chuyên gia toán học. Chỉ tập trung giải toán.")

def assistant(state: MessagesState):
    # Kẹp [sys_msg] lên đầu danh sách tin nhắn hiện tại
    return {"messages": [llm.invoke([sys_msg] + state["messages"])]}
```

---

## 7. Các công cụ xây sẵn (Pre-built) của LangGraph

Để không phải viết đi viết lại những đoạn code nhàm chán, LangGraph cung cấp sẵn các tiện ích tích hợp:

**`ToolNode` — Xưởng thực thi tự động:** Tự động nhận lệnh từ AI, lôi hàm Python ra chạy, và đóng gói kết quả trả về State.
```python
from langgraph.prebuilt import ToolNode

builder.add_node("tools", ToolNode(tools))  # Truyền danh sách công cụ vào đây
```

**`tools_condition` — Cảnh sát giao thông:** Hàm kiểm tra xem AI vừa trả lời cái gì để phân luồng.

- Nếu AI gọi tool ➡️ Rẽ vào trạm `"tools"`.
- Nếu AI trả lời bằng văn bản bình thường ➡️ Rẽ ra cổng `END`.
```python
from langgraph.prebuilt import tools_condition

builder.add_conditional_edges("assistant", tools_condition)
```

---

## 8. Cấp Trí nhớ cho Agent (Memory & Checkpointer)

Mặc định, LangGraph mắc bệnh **"não cá vàng"** — quên hết mọi thứ sau khi chạy xong. Để nó có thể trò chuyện dài hơi, ta cần thêm cơ chế lưu trữ.

- **Checkpointer:** Giống như một cái **Tủ Hồ Sơ**. Nó tự động chụp ảnh (snapshot) lại State sau mỗi bước chạy.
- **`thread_id`:** Mã số của cuộc hội thoại — để AI phân biệt người này với người khác.
```python
from langgraph.checkpoint.memory import MemorySaver

# Khởi tạo Tủ Hồ Sơ (lưu vào RAM)
memory = MemorySaver()

# Giao tủ cho đồ thị khi compile
react_agent = builder.compile(checkpointer=memory)

# Khi gọi AI, phải đưa kèm "Thẻ tên" (thread_id)
config = {"configurable": {"thread_id": "phong_chat_cua_an"}}
react_agent.invoke({"messages": [("user", "Tôi tên là An")]}, config)
```

---

## 🌟 9. Bộ Khung Hoàn Chỉnh (Master Template)

Chỉ với đoạn code này, bạn đã sở hữu một AI Agent có tư duy, có khả năng dùng công cụ và có trí nhớ:
```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage

# 1. State & Tools
class State(TypedDict):
    messages: Annotated[list, add_messages]

tools = [cong_cu_1, cong_cu_2]
llm_with_tools = llm.bind_tools(tools)
sys_msg = SystemMessage(content="Bạn là một trợ lý ảo siêu việt.")

# 2. Định nghĩa Node AI
def assistant(state: State):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# 3. Xây dựng đồ thị
builder = StateGraph(State)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# 4. Phân luồng giao thông (Routing)
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")  # Vòng lặp ReAct

# 5. Đóng gói với Trí nhớ
memory = MemorySaver()
app = builder.compile(checkpointer=memory)

# 6. Sử dụng
config = {"configurable": {"thread_id": "demo_1"}}
response = app.invoke({"messages": [("user", "Hãy làm nhiệm vụ này...")]}, config)
```
