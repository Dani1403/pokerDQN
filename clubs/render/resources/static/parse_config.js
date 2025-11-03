(function init() {
    const socket = io();

    socket.on("connect", () => {
        console.log("[SOCKET] Connected to server.");
    });

    socket.on("config", function (config) {
        console.log("[SOCKET] Received config:", config);
        reset_player();
        reset_community();
        reset_button();

        if (config.hasOwnProperty("hole_cards")) {
            update_players(config);
            update_community(config);
            update_button(config);
            update_action(config);
        }
    });

    function reset_player() {
        let players = document.getElementsByClassName("player");
        for (let player_idx = 0; player_idx < players.length; player_idx++) {
            let player = players[player_idx];
            const chips = document.getElementById(`chips-text-${player_idx}`);
            if (chips) chips.innerHTML = 0;

            let card_backgrounds = player.getElementsByClassName("card-background");
            for (let bg of card_backgrounds) {
                bg.setAttribute("fill", "url(#card-back)");
            }

            let card_texts = player.getElementsByClassName("card-text");
            for (let txt of card_texts) {
                txt.innerHTML = "";
            }
        }

        let player_backgrounds = document.getElementsByClassName("player-background");
        for (let bg of player_backgrounds) {
            bg.setAttribute("fill", "#ffffff");
        }
    }

    function reset_button() {
        let buttons = document.getElementsByClassName("button-background");
        for (let btn of buttons) {
            btn.setAttribute("fill", "transparent");
        }
    }

    function reset_community() {
        let community = document.getElementById("community");
        if (!community) return;

        let card_backgrounds = community.getElementsByClassName("card-background");
        if (card_backgrounds[0])
            card_backgrounds[0].setAttribute("fill", "url(#card-back)");
        for (let i = 1; i < card_backgrounds.length; i++) {
            card_backgrounds[i].setAttribute("fill", "url(#card-blank)");
        }

        let card_texts = community.getElementsByClassName("card-text");
        if (card_texts[0])
            card_texts[0].setAttribute("fill", "url(#card-back)");
        for (let i = 1; i < card_texts.length; i++) {
            card_texts[i].innerHTML = "";
        }

        const pot = document.getElementById("pot-text");
        if (pot) pot.innerHTML = 0;
    }

    function update_players(config) {
        for (let player_idx = 0; player_idx < config["hole_cards"].length; player_idx++) {
            const chips = document.getElementById(`chips-text-${player_idx}`);
            if (chips) chips.innerHTML = config["stacks"][player_idx];

            const commit = document.getElementById(`street-commit-text-${player_idx}`);
            if (commit) commit.innerHTML = config["street_commits"][player_idx];

            if (config["active"][player_idx]) {
                let cards = config["hole_cards"][player_idx];
                for (let card_idx = 0; card_idx < cards.length; card_idx++) {
                    let card = document.getElementById(`card-${player_idx}-${card_idx}`);
                    let card_string = cards[card_idx];
                    if (card) update_card(card, card_string);
                }
            }
        }
    }

    function update_card(card, card_string) {
        if (!card_string || card_string.length < 2) return;

        let value = card_string[0];
        if (value === "T") value = "10";

        let text = card.getElementsByClassName("card-text")[0];
        if (text) text.innerHTML = value;

        let suit = card_string[1];
        let background = card.getElementsByClassName("card-background")[0];

        if (!background || !text) return;

        switch (suit) {
            case "♣":
                background.setAttribute("fill", "url(#card-club)");
                text.setAttribute("stroke", "black");
                text.setAttribute("fill", "black");
                break;
            case "♠":
                background.setAttribute("fill", "url(#card-spade)");
                text.setAttribute("stroke", "black");
                text.setAttribute("fill", "black");
                break;
            case "♥":
                background.setAttribute("fill", "url(#card-heart)");
                text.setAttribute("stroke", "red");
                text.setAttribute("fill", "red");
                break;
            case "♦":
                background.setAttribute("fill", "url(#card-diamond)");
                text.setAttribute("stroke", "red");
                text.setAttribute("fill", "red");
                break;
        }
    }

    function update_community(config) {
        let cards = config["community_cards"];
        for (let i = 0; i < cards.length; i++) {
            let card = document.getElementById(`card-community-${i + 1}`);
            if (card) update_card(card, cards[i]);
        }

        const pot = document.getElementById("pot-text");
        if (pot) pot.innerHTML = config["pot"];
    }

    function update_button(config) {
        let btn = document.getElementById(`button-background-${config["button"]}`);
        if (btn) btn.setAttribute("fill", "url(#dealer)");
    }

    function update_action(config) {
        if (config["action"] >= 0) {
            let bg = document.getElementById(`player-background-${config["action"]}`);
            if (bg) bg.setAttribute("fill", "#000000");
        }
    }
})();
