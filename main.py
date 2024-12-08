import json
import random
from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image, ImageTk
import requests
from io import BytesIO
import tkinter as tk
from tkinter import messagebox, ttk


# BERT 모델 및 토크나이저 초기화
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model = AutoModel.from_pretrained("bert-base-multilingual-cased")


def load_pokedex(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return torch.mean(outputs.last_hidden_state, dim=1).squeeze()


def get_pokemon_feature(pokemon):
    features = [
        str(pokemon['id']),
        pokemon['name'],
        ", ".join(pokemon['type']),
        pokemon['height'],
        pokemon['weight'],
        pokemon['candy'],
        pokemon['egg'] if pokemon['egg'] else "",
        str(pokemon['multipliers']) if pokemon['multipliers'] else "",
        ", ".join(pokemon['weaknesses']),
        ", ".join([evo['name'] for evo in pokemon.get('prev_evolution', [])]) if pokemon.get('prev_evolution') else "",
        ", ".join([evo['name'] for evo in pokemon.get('next_evolution', [])]) if pokemon.get('next_evolution') else "",
    ]
    return ", ".join(features)


class PokemonGameApp:
    def __init__(self, root, pokedex):
        self.root = root
        self.root.title("포켓몬 고 유사도 게임")
        self.pokedex = pokedex
        self.target_pokemon = random.choice(pokedex)
        self.target_embedding = embed_text(get_pokemon_feature(self.target_pokemon))
        self.rankings = []  # 랭킹 데이터를 저장
        self.incorrect_attempts = 0

        self.setup_ui()

    def setup_ui(self):
        # 포켓몬 이름 또는 타입 입력
        self.label = tk.Label(self.root, text="포켓몬 이름 또는 타입 입력:", font=("Arial", 14))
        self.label.pack(pady=10)

        self.entry = tk.Entry(self.root, font=("Arial", 14), width=30)
        self.entry.pack(pady=10)

        self.submit_button = tk.Button(self.root, text="확인", font=("Arial", 12), command=self.check_pokemon)
        self.submit_button.pack(pady=5)

        self.show_rankings_button = tk.Button(self.root, text="랭킹 보기", font=("Arial", 12), command=self.show_rankings)
        self.show_rankings_button.pack(pady=5)

        self.exit_button = tk.Button(self.root, text="종료", font=("Arial", 12), command=self.root.quit)
        self.exit_button.pack(pady=5)

        # 포켓몬 정보 표시
        self.info_frame = tk.Frame(self.root)
        self.info_frame.pack(pady=20)

        self.info_label = tk.Label(self.info_frame, text="포켓몬 정보:", font=("Arial", 14))
        self.info_label.grid(row=0, column=0, sticky="w")

        self.info_text = tk.Text(self.info_frame, width=60, height=15, font=("Arial", 12))
        self.info_text.grid(row=1, column=0, pady=10)

        # 이미지 표시
        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=10)

    def check_pokemon(self):
        input_name = self.entry.get().strip()
        if not input_name:
            messagebox.showerror("입력 오류", "포켓몬 이름 또는 타입을 입력하세요.")
            return

        # 타입으로 검색
        matching_pokemons = [p for p in self.pokedex if input_name in p['type']]
        if matching_pokemons:
            self.display_type_results(matching_pokemons)
            return

        # 이름으로 검색
        matched_pokemon = next((p for p in self.pokedex if p['name'] == input_name), None)
        if not matched_pokemon:
            messagebox.showerror("입력 오류", "해당 이름의 포켓몬이 없습니다.")
            return

        # 유사도 계산
        input_embedding = embed_text(get_pokemon_feature(matched_pokemon))
        similarity = torch.cosine_similarity(input_embedding, self.target_embedding, dim=0).item() * 100

        # 포켓몬 정보 표시
        self.display_pokemon_info(matched_pokemon, similarity)

        # 랭킹 추가
        self.rankings.append((matched_pokemon['name'], similarity))
        self.rankings.sort(key=lambda x: x[1], reverse=True)  # 유사도 순 정렬

        # 정답 여부 확인
        if matched_pokemon['name'] == self.target_pokemon['name']:
            messagebox.showinfo("정답", "축하합니다! 정답을 맞췄습니다!")
        else:
            self.incorrect_attempts += 1
            if self.incorrect_attempts % 5 == 0:
                self.provide_hint()

    def display_type_results(self, matching_pokemons):
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "입력한 타입의 포켓몬 목록:\n")
        for pokemon in matching_pokemons:
            self.info_text.insert(tk.END, f"- {pokemon['name']}\n")

    def display_pokemon_info(self, pokemon, similarity):
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"입력한 포켓몬: {pokemon['name']}\n")
        self.info_text.insert(tk.END, f"ID: {pokemon['id']}\n")
        self.info_text.insert(tk.END, f"타입: {', '.join(pokemon['type'])}\n")
        self.info_text.insert(tk.END, f"키: {pokemon['height']}\n")
        self.info_text.insert(tk.END, f"몸무게: {pokemon['weight']}\n")
        self.info_text.insert(tk.END, f"유사도: {similarity:.2f}%\n")

        self.display_pokemon_image(pokemon["img"])

    def display_pokemon_image(self, img_url):
        response = requests.get(img_url)
        img_data = Image.open(BytesIO(response.content)).resize((150, 150))
        img = ImageTk.PhotoImage(img_data)

        self.img_label.config(image=img)
        self.img_label.image = img

    def provide_hint(self):
        hints = [
            ("키", self.target_pokemon["height"]),
            ("몸무게", self.target_pokemon["weight"]),
            ("알", self.target_pokemon["egg"] if self.target_pokemon["egg"] else "없음"),
            ("약점", random.choice(self.target_pokemon["weaknesses"])),
        ]
        hint = random.choice(hints)
        messagebox.showinfo("힌트", f"{hint[0]}: {hint[1]}")

    def show_rankings(self):
        # 새로운 창 생성
        rankings_window = tk.Toplevel(self.root)
        rankings_window.title("랭킹")
        rankings_window.geometry("400x400")

        # 랭킹 출력
        rankings_label = tk.Label(rankings_window, text="랭킹", font=("Arial", 16))
        rankings_label.pack(pady=10)

        rankings_text = tk.Text(rankings_window, font=("Arial", 12), width=40, height=20)
        rankings_text.pack(pady=10)

        for rank, (name, similarity) in enumerate(self.rankings, start=1):
            rankings_text.insert(tk.END, f"{rank}. {name}: {similarity:.2f}%\n")


if __name__ == "__main__":
    pokedex_file = "./pogo-mantle/pokedex.json"
    pokedex = load_pokedex(pokedex_file)

    root = tk.Tk()
    app = PokemonGameApp(root, pokedex)
    root.mainloop()
