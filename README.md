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
