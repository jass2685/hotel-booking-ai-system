# IMPORTING ALL LIBRARIES
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample fallback hotels (used if hotels.csv not present)
sample_hotels = pd.DataFrame({
    "City": [
        "Patiala","Patiala","Patiala","Patiala","Patiala",
        "Chandigarh","Chandigarh","Chandigarh","Chandigarh","Chandigarh",
        "Rajpura","Rajpura","Rajpura","Rajpura","Rajpura",
    ],
    "Hotel": [
        "Hotel Heritage", "Royal Residency", "Patiala Palace","Ran Baas The Palace","The Baradari Palace",
        "City Star", "The Orchid", "Hotel Midtown","Golden Plaza Hotel& Spa","Hyatt Centric",
        "Raj Classic", "The Grand Raj","Hotel Icon","Hotel Amar Palace","The Firangipani Country",
    ],
    "Price": [2200, 2600, 2800,10000 ,8000,3200, 4000, 2700, 8500,10000,2400, 4000,5000,4400,7500],
    "Stars": [3, 4, 4, 5, 5, 3, 3, 4,5,5,3,3,2,4,5],
    "Rating": [4.1, 4.3, 4.0,5, 4.9, 3.9, 3.4, 4,4.9,5,3.3,4.2,2.9,4,4.8],
    "Facilities": [
        ["WiFi","Breakfast","Gym"],
        ["WiFi","Breakfast","Dinner","Gym"],
        ["WiFi","Breakfast","Lunch","Dinner"],
        ["WiFi","Breakfast","Lunch","Dinner","Pool","Gym","spa","Parking"],
        ["WiFi","Breakfast","Dinner","Pool","gym","Parking"],
        ["WiFi","Breakfast","Gym"],
        ["wifi","Breakfast","Dinner"],
        ["Lunch","Gym","Parking"],
        ["WiFi","Breakfast","Lunch","Dinner","Pool","Gym","spa","Parking"],
        ["WiFi","Breakfast","Dinner","Pool","gym","Parking"],
        ["wifi","Breakfast","Dinner"],
        ["Lunch","Dinner","Parking"],
        ["Lunch","Parking"],
        ["Lunch","Dinner","Parking"],
        ["WiFi","Breakfast","Dinner","Pool","Parking"],
    ],
})
# FUNCTION LOAD HOTEL DATA
def load_hotels():
    if os.path.exists("hotels.csv"):
        df = pd.read_csv("hotels.csv")
        df["Facilities"] = df["Facilities"].apply(lambda x: [s.strip() for s in str(x).split(";")])
        # convert numeric columns if necessary
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).astype(int)
        df["Stars"] = pd.to_numeric(df["Stars"], errors="coerce").fillna(3).astype(int)
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(3.0)
        return df
    else:
        return sample_hotels.copy()

hotels = load_hotels()
# Helper: rebuild ML features when hotels/reviews/bookings change
mlb = None
hotel_features = None

def rebuild_features():
    global mlb, hotel_features
    mlb = MultiLabelBinarizer()
    facilities_encoded = mlb.fit_transform(hotels["Facilities"])
    # normalize price, stars, rating (z-score)
    eps = 1e-6
    price = hotels["Price"].astype(float).values.reshape(-1,1)
    if price.std() < eps: price_norm = np.zeros_like(price)
    else: price_norm = (price - price.mean()) / (price.std())
    stars = hotels["Stars"].astype(float).values.reshape(-1,1)
    stars_norm = (stars - stars.mean()) / (stars.std() + eps)
    rating = hotels["Rating"].astype(float).values.reshape(-1,1)
    rating_norm = (rating - rating.mean()) / (rating.std() + eps)
    # bookings count influence
    if os.path.exists("bookings.csv"):
        bc_df = pd.read_csv("bookings.csv")
        counts = bc_df["Hotel"].value_counts()
        bc = hotels["Hotel"].apply(lambda h: np.log1p(int(counts.get(h,0)))).values.reshape(-1,1)
    else:
        bc = np.zeros((len(hotels),1))
    hotel_features = np.hstack([facilities_encoded, price_norm, stars_norm, rating_norm, bc])

rebuild_features()
# Recommendation logic
def recommend_best(city, preferred_facilities):
    subset_idx = hotels[hotels["City"].str.lower() == city.lower()].index
    if len(subset_idx) == 0:
        return None
    # build user vector in same order as mlb.classes_
    user_fac = [f.strip() for f in preferred_facilities if f.strip()]
    user_fac_vec = mlb.transform([user_fac]) if len(user_fac)>0 else np.zeros((1,len(mlb.classes_)))
    # add typical numeric pref (use average of hotels in city)
    city_prices = hotels.loc[subset_idx, "Price"].values
    city_stars = hotels.loc[subset_idx, "Stars"].values
    city_rating = hotels.loc[subset_idx, "Rating"].values
    # use medians to create user numeric vector
    price_val = np.median(city_prices) if len(city_prices)>0 else hotels["Price"].median()
    stars_val = np.median(city_stars) if len(city_stars)>0 else hotels["Stars"].median()
    rating_val = np.median(city_rating) if len(city_rating)>0 else hotels["Rating"].median()
    # normalize using hotels overall stats (same as rebuild_features)
    # we will compute normalized single-row numeric vector with same transforms
    # get price_norm etc using hotels stats:
    price_all = hotels["Price"].astype(float)
    stars_all = hotels["Stars"].astype(float)
    rating_all = hotels["Rating"].astype(float)
    eps=1e-6
    price_norm = (price_val - price_all.mean()) / (price_all.std() + eps)
    stars_norm = (stars_val - stars_all.mean()) / (stars_all.std() + eps)
    rating_norm = (rating_val - rating_all.mean()) / (rating_all.std() + eps)
    # bookings = 0 for user
    bc_val = 0.0
    user_vec = np.hstack([user_fac_vec, [[price_norm, stars_norm, rating_norm, bc_val]]])
    # compute similarities only with subset rows
    sim = cosine_similarity(user_vec, hotel_features[subset_idx].astype(float))
    best_idx_rel = int(sim.argmax())
    best_idx = subset_idx[best_idx_rel]
    return hotels.loc[best_idx]
# UI and app logic
root = tk.Tk()
root.title("🏨 HOTEL BOOKING")
root.geometry("2000x16000")
root.configure(bg="#eef6fb")

title = tk.Label(root, text="🤖 Hotel Booking ", font=("times new roman",50,'bold', "bold"), bg="#eef6fb", fg="#1894c9",relief="ridge")
title.pack(fill="x",pady=8)

# top frame: name, city, show button
top_frame = tk.Frame(root, bg="#eef6fb")
top_frame.pack(pady=10,anchor="w",padx=20)
tk.Label(top_frame, text="Name:",font=("Times New Roman", 14, "bold"), bg="#eef6fb").grid(row=0,column=0, padx=6)
name_var = tk.StringVar()
tk.Entry(top_frame, textvariable=name_var, width=70).grid(row=0,column=1, padx=(6,50))

tk.Label(top_frame, text="City:",font=("Times New Roman", 14, "bold"), bg="#eef6fb").grid(row=0,column=2, padx=6)
city_var = tk.StringVar()
city_entry = tk.Entry(top_frame, textvariable=city_var, width=70)
city_entry.grid(row=0,column=3, padx=(6,50))

show_btn = tk.Button(top_frame, text="🔍 Show Hotels", bg="#2e86c1", fg="white",font=("Times New Roman",14,"bold"),width=18, command=lambda: show_hotels())
show_btn.grid(row=0, column=4, padx=14,pady=5)

# middle: preferred facilities + recommend AI
pref_frame = tk.Frame(root, bg="#eef6fb")
pref_frame.pack(pady=12)
tk.Label(pref_frame, text="Preferred Facilities (comma separated):",font=("Times New Roman",14,"bold"), bg="#eef6fb").grid(row=0,column=0, sticky="w")
pref_var = tk.StringVar()
tk.Entry(pref_frame, textvariable=pref_var, width=60).grid(row=0,column=1, padx=6)
rec_btn = tk.Button(pref_frame, text="🤖 Recommend Best Hotel",font=("Times New Roman",16,"bold"), bg="#27ae60", fg="white",command=lambda: recommend_ai())
rec_btn.grid(row=0,column=2, padx=(10,50))

# results tree
style = ttk.Style()
style.theme_use("default")

style.configure(
    "Treeview",
    font=("Arial", 12),
    rowheight=30,
    background="#ffffff",
    fieldbackground="#ffffff",
    foreground="#000000"
)

style.configure(
    "Treeview.Heading",
    font=("Times New Roman", 14, "bold"),
    background="#004080",
    foreground="white"
)

style.map(
    "Treeview",
    background=[("selected", "#99ccff")],
    foreground=[("selected", "black")]
)

columns = ("Hotel","City","Price","Rating","Stars")
result_box = ttk.Treeview(root, columns=columns, show="headings", height=8)
for col in columns:
    result_box.heading(col, text=col)
    result_box.column(col, anchor="center", width=150)
result_box.pack(pady=10, fill="x", padx=10)
result_box.tag_configure("oddrow", background="#f2f2f2")
result_box.tag_configure("evenrow", background="#ffffff")

# booking / review area
bt_frame = tk.Frame(root, bg="#eef6fb")
bt_frame.pack(pady=6)
book_btn = tk.Button(bt_frame, text="📘 Book Selected Hotel",font=("Times New Roman",14,"bold"), bg="#4f0755", fg="white", width=20, command=lambda: book_hotel())
book_btn.grid(row=0,column=0, padx=8)
view_booking_btn = tk.Button(bt_frame, text="📄 View Bookings", font=("Times New Roman",14,"bold"),bg="#f39c12", fg="white", width=20, command=lambda: view_bookings())
view_booking_btn.grid(row=0,column=1, padx=8)

# Review area
rev_frame = tk.LabelFrame(root, text="Leave a Review for Selected Hotel",font=("Times New Roman",12,"bold"),bg="#eef6fb")
rev_frame.pack(pady=8, padx=10, fill="x")
tk.Label(rev_frame, text="Rating (0.5 - 5):",font=("Times New Roman",12,"bold"), bg="#eef6fb").grid(row=0,column=0, padx=6, pady=4)
review_rating_var = tk.StringVar()
tk.Entry(rev_frame, textvariable=review_rating_var, width=8).grid(row=0,column=1, padx=6)
tk.Label(rev_frame, text="Comment:",font=("Times New Roman",12,"bold"),bg="#eef6fb").grid(row=0,column=2)
review_comment_text = tk.Text(rev_frame, height=5, width=90)
review_comment_text.grid(row=1,column=0,columnspan=5, padx=6, pady=4)
submit_rev_btn = tk.Button(rev_frame, text="✍️ Submit Review",font=("Times New Roman",10,"bold"), bg="#16a085", fg="white", command=lambda: submit_review())
submit_rev_btn.grid(row=0,column=3, padx=8, pady=4)

# helper messages
msg = tk.Label(root, text="Tip: For Facilities use names like WiFi, Breakfast, Dinner, Pool, Gym, Parking,spa", font=("Times New Roman",12,"bold"),bg="#eef6fb")
msg.pack(pady=6)
# Functions used by UI
def show_hotels():
    city = city_var.get().strip()
    result_box.delete(*result_box.get_children())
    if not city:
        messagebox.showwarning("Input required", "Please enter a city name.")
        return
    res = hotels[hotels["City"].str.lower() == city.lower()]
    if res.empty:
        messagebox.showinfo("No hotels", f"No hotels found in {city}.")
        return
    for _, r in res.iterrows():
        result_box.insert("", "end", values=(r["Hotel"], r["City"], f"₹{r['Price']}", r["Rating"], f"{r['Stars']}-Star"))

def book_hotel():
    sel = result_box.focus()
    if not sel:
        messagebox.showwarning("Select Hotel", "Please select a hotel to book.")
        return
    vals = result_box.item(sel, "values")
    hotel_name, city, price, rating, stars = vals
    user = name_var.get().strip()
    if not user:
        messagebox.showwarning("Name required", "Please enter your name to book.")
        return
    row = [user, hotel_name, city, price, rating, stars, datetime.now().isoformat()]
    df = pd.DataFrame([row], columns=["User","Hotel","City","Price","Rating","Stars","BookedAt"])
    if os.path.exists("bookings.csv"):
        df.to_csv("bookings.csv", mode="a", header=False, index=False)
    else:
        df.to_csv("bookings.csv", index=False)
    messagebox.showinfo("Booked", f"Booking confirmed for {hotel_name}.")
    rebuild_features()

def view_bookings():
    if not os.path.exists("bookings.csv"):
        messagebox.showinfo("No Bookings", "No bookings yet.")
        return
    df = pd.read_csv("bookings.csv")
    win = tk.Toplevel(root)
    win.title("Bookings")
    win.geometry("800x400")
    tree = ttk.Treeview(win, columns=list(df.columns), show="headings")
    for c in df.columns:
        tree.heading(c, text=c)
        tree.column(c, width=110)
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row.values))
    tree.pack(expand=True, fill="both")

def submit_review():
    sel = result_box.focus()
    if not sel:
        messagebox.showwarning("Select hotel", "Select a hotel to review.")
        return
    hotel_name = result_box.item(sel, "values")[0]
    rating = review_rating_var.get().strip()
    comment = review_comment_text.get("1.0","end").strip()
    try:
        r = float(rating)
        if r < 0.5 or r > 5:
            raise ValueError
    except:
        messagebox.showerror("Invalid", "Rating must be a number between 0.5 and 5.")
        return
    row = [hotel_name, r, comment, datetime.now().isoformat()]
    df = pd.DataFrame([row], columns=["Hotel","Rating","Comment","When"])
    if os.path.exists("reviews.csv"):
        df.to_csv("reviews.csv", mode="a", header=False, index=False)
    else:
        df.to_csv("reviews.csv", index=False)
    messagebox.showinfo("Thanks", "Review saved.")
    review_rating_var.set("")
    review_comment_text.delete("1.0","end")
    refresh_ratings_from_reviews()

def refresh_ratings_from_reviews():
    if not os.path.exists("reviews.csv"):
        return
    rv = pd.read_csv("reviews.csv")
    avg = rv.groupby("Hotel")["Rating"].mean().to_dict()
    for idx, row in hotels.iterrows():
        if row["Hotel"] in avg:
            # simple average of original rating and review mean
            hotels.at[idx, "Rating"] = round((row["Rating"] + avg[row["Hotel"]]) / 2, 2)
    rebuild_features()

def recommend_ai():
    city = city_var.get().strip()
    pref = [f.strip() for f in pref_var.get().split(",") if f.strip()]
    if not city:
        messagebox.showwarning("Missing", "Enter city name first.")
        return
    subset_idx = hotels[hotels["City"].str.lower() == city.lower()].index
    if len(subset_idx) == 0:
        messagebox.showinfo("No Hotels Found", "City not found in the database.")
        return

    # build user vector
    user_fac = [f.strip() for f in pref if f.strip()]
    user_fac_vec = mlb.transform([user_fac]) if len(user_fac)>0 else np.zeros((1,len(mlb.classes_)))
    price_val = hotels["Price"].median()
    stars_val = hotels["Stars"].median()
    rating_val = hotels["Rating"].median()

    eps=1e-6
    price_norm = (price_val - hotels["Price"].mean()) / (hotels["Price"].std() + eps)
    stars_norm = (stars_val - hotels["Stars"].mean()) / (hotels["Stars"].std() + eps)
    rating_norm = (rating_val - hotels["Rating"].mean()) / (hotels["Rating"].std() + eps)
    user_vec = np.hstack([user_fac_vec, [[price_norm, stars_norm, rating_norm, 0.0]]])

    sim = cosine_similarity(user_vec, hotel_features[subset_idx].astype(float))
    top_indices = sim[0].argsort()[-3:][::-1]  
    top_hotels = hotels.iloc[subset_idx[top_indices]]
    result = "🏆 Top Recommended Hotels in " + city + ":\n\n"
    for i, row in top_hotels.iterrows():
        result += f"🏨 {row['Hotel']}\n💰 ₹{row['Price']}\n⭐ {row['Stars']} stars\n🌟 Rating: {row['Rating']}\nFacilities: {', '.join(row['Facilities'])}\n\n"

    messagebox.showinfo("🤖 AI Recommendations", result)
    
# initial rating refresh (if reviews exist)
refresh_ratings_from_reviews()

root.mainloop()
